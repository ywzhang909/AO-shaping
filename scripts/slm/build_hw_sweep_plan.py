#!/usr/bin/env python
"""生成「硬件离焦扫描 + 多阶泽尼克匀化」第三轮的采集计划 JSON。

计划由 ``scripts/slm/run_hw_sweep_groups.py`` 读: 每一组 = 一次独立子进程调用。

为什么要有这个脚本
------------------

上一轮 (第二轮) 的计划是**手写**的 JSON (``data/debug/hw_groups_20261021/plan.json``,
26 组)。手写的代价在第三轮立刻兑现:

* 焦距从 1626.2 mm 变成 1523.8 mm 时, 20 处 ``--defocus-focal-mm`` 靠人逐个改;
* 第三轮还要同时改**数据根**、**块数**、**路径** —— 手改必然漏, 而漏掉的地方
  **不报错**: 上一轮逐阶组失败后, 离焦组静默用了**上一轮遗留**的相位文件,
  数据看起来齐全, 其实是两轮混在一起。

所以这一轮把计划改成**脚本生成**: 每一项可变量 (焦距 / 曝光 / 块数 / 总轮数 /
每块轮数 / 墙钟预算 / 数据根 / sweep 根 / Δz 点表 / 孔径 …) 都是命令行参数,
生成时**自检**五件事, 不过就报错退出, 不写出一份错的计划。

第三轮与上一轮的三处差异
------------------------

1. **新的数据根**。上一轮与第二轮共用 ``data/debug/hw_groups_20261021/`` 与
   ``data/debug/hw_zernike_sweep_20261021/``, 于是第三轮的离焦/远场组
   (15:57–16:01) 把第二轮那 11 个组的 ``measurements.json`` **覆盖**了 ——
   第二轮报告里的这些数字在磁盘上已不可复现。这一轮所有 ``--out-root`` /
   ``--out`` / ``--stage-a-in`` / 相位文件路径都指向**新根**。
   ⚠️ 上一轮计划里离焦组与远场组引用的相位文件写的是**旧根**, 于是逐阶组失败后
   离焦组静默用了上一轮遗留的相位文件而不报错 —— 这一轮的自检 (c) 就是堵这个坑。

2. **块数 6 → 7**。Stage A 每块有墙钟预算 ``--time-budget-s 540``, 实测第一块
   549.3 s 只跑出 **39 轮** (不是请求的 45) ⇒ 每块实际产出 ≈ 39 轮,
   6 块 × 39 ≈ 234 轮 **够不到 250 轮**的总目标 ⇒ 7 块。
   ⚠️ 39 是墙钟截断的**结果**, 不是「每块只跑 39 轮」: 请求值仍是 45
   (``--chunk-epochs``), 扫描脚本自己用 ``min(chunk_epochs, epochs - 已跑轮数)``
   给最后一块收缩, 所以最后一块自动跑剩余量。

3. **其余沿用上一轮的形状**: 组序、每组的自变量、Stage A 的全部参数、离焦与
   远场组的点表与孔径都照抄, 只把路径换到新根。远场组按阶数**升序**排 (上一轮
   留下的 11,15,20,25,3,4,6,8 是手写乱序)。

用法
----

::

    # 只看生成结果与自检报告, 不落盘
    python scripts/slm/build_hw_sweep_plan.py --print-plan

    # 写计划 (默认写到 <groups-root>/plan.json)
    python scripts/slm/build_hw_sweep_plan.py

    # 换焦距 / 换根 / 改块数预算
    python scripts/slm/build_hw_sweep_plan.py --defocus-focal-mm 1600.0 \\
        --groups-root data/debug/hw_groups_20261022 \\
        --sweep-root data/debug/hw_zernike_sweep_20261022

    # 跑它 (驱动会逐组开独立子进程)
    python scripts/slm/run_hw_sweep_groups.py \\
        --plan data/debug/hw_groups_20261022/plan.json --dry-run
"""

from __future__ import annotations

import argparse
from loguru import logger
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 默认值 (模块级常量 —— 每一项都能被命令行覆盖)
# ---------------------------------------------------------------------------
DEFAULT_GROUPS_ROOT = "data/debug/hw_groups_20261022"
"""本轮**分组**根目录 (离焦/远场组与焦面标定证据落在这里)。"""

DEFAULT_SWEEP_ROOT = "data/debug/hw_zernike_sweep_20261022"
"""本轮**泽尼克扫描**根目录 (Stage A / 逐阶组落在这里, 相位文件也从这里读)。"""

DEFAULT_DEFOCUS_FOCAL_MM = 1523.8
"""瞳面菲涅尔焦面矫正焦距 (mm)。上一轮实测值, 换根时必须一起换。"""

DEFAULT_TOTAL_EPOCHS = 250
"""Stage A 的**总**精化轮数 (学习率调度的总地平线, 与分块数无关)。"""

DEFAULT_CHUNK_EPOCHS = 45
"""每块**请求**的轮数。墙钟预算只管得住实际产出, 请求值仍按这个写。"""

DEFAULT_EPOCHS_PER_CHUNK = 39
"""一块在墙钟预算内**实际**能跑出的轮数 (2026-10-21 实测: 549.3 s ⇒ 39 轮)。

块数由它算: ``ceil(total_epochs / epochs_per_chunk)``。默认 250/39 ⇒ 7 块。
"""

DEFAULT_TIME_BUDGET_S = 540.0
"""Stage A 每块的墙钟预算 (s)。实测 45 轮要 ~549 s, 所以 540 s 会截断到 ~39 轮。"""

DEFAULT_ORDERS = (3, 4, 6, 8, 11, 15, 20, 25)
"""逐阶组与远场补拍组扫的泽尼克截断阶数 (**升序**)。"""

DEFAULT_DZ_NOISE_MM = (0.0,) * 9
"""噪声组: 同一个 Δz 重复 9 次, 用来量同一光斑的重复性。"""

DEFAULT_DZ_SCAN_MM = (
    -20.0, -15.0, -10.0, -7.5, -5.0, -3.75, -2.5, -1.5007, -0.7504, -0.3752,
    -0.1876, 0.0, 0.1876, 0.3752, 0.7504, 1.5007, 2.5, 3.75, 5.0, 7.5, 10.0,
    15.0, 20.0,
)
"""离焦扫描点 (mm)。0 附近是几何级数加密 —— 离焦响应在 0 两侧不对称。"""

DEFAULT_DZ_SCAN_MODEL_MM = DEFAULT_DZ_SCAN_MM + (25.0, 30.0, 35.0, 40.0, 45.0, 50.0)
"""模型相位对照组的扫描点: 比硬件组多 6 个大 Δz 点, 让两条曲线在 20 mm 处交叠。"""

DEFAULT_HW_TARGET_SIDE_PX = 10
"""硬件相位组的目标框边长 (CCD px) = 平场腰径 2·w0。"""

DEFAULT_MODEL_TARGET_SIDE_PX = 44
"""模型相位组的目标框边长 (CCD px) = 模型方斑边长。

🔴 **孔径是预先登记的, 别改**: 第一轮把 44 px 用在硬件相位上, 离焦响应被口径
藏住了 (框比光斑大得多 ⇒ 框内能量几乎不随离焦变)。硬件 10 px / 模型 44 px
是**两套相位各自的定尺**, 不是可以统一的参数。
"""

DEFAULT_STAGE_A_EXPOSURE_MS = 0.2
DEFAULT_DZ_HW_EXPOSURE_MS = 0.2
DEFAULT_DZ_MODEL_EXPOSURE_MS = 0.4
DEFAULT_FARFIELD_EXPOSURE_MS = 0.2
DEFAULT_FOCUS_EXPOSURE_MS = 0.1
DEFAULT_FOCUS_FRAMES = 3
DEFAULT_FOCUS_MAX_EXPOSURE_MS = 20.0
DEFAULT_FOCUS_STEPS = 33
DEFAULT_FOCUS_DIOPTER_RANGE = 6.0
DEFAULT_FOCUS_REFINE = 9
DEFAULT_FRAMES = 4

DEFAULT_CAM_TYPE = "daheng"
DEFAULT_CAM_ID = 0
DEFAULT_CAM_SIZE = 256
DEFAULT_BEAM_RADIUS_PX = 450.0
DEFAULT_SLM_NUMBER = 1
DEFAULT_GS_ITERS = 200
DEFAULT_SPGD_EPOCHS = 150
DEFAULT_LR = 0.01
DEFAULT_DELTA = 0.02
DEFAULT_MAX_ENERGY_LOSS = 0.6
DEFAULT_SEED = 0

DEFAULT_PLAN_EXPOSURE_CAP_MS = 0.4
"""计划级曝光上限。内置的加倍阶梯估计在暗实验系统上会给出假上限, 必须写死。"""

DEFAULT_FOCUS_EXPOSURE_CAP_MS = 40.0
"""焦面标定组的曝光上限覆盖: 它要一路加曝光到 20 ms 扫阶梯。"""

#: 计划里用到的两个脚本 (相对仓库根)。
STAGE_A_SCRIPT = "scripts/slm/run_gs_refine_zernike_sweep.py"
DEFOCUS_SCRIPT = "scripts/slm/hw_gs_defocus_probe.py"

#: 焦面标定是**模块**不是脚本, 计划里用 ``script: "-m"`` 表示。
FOCUS_PROBE_MODULE = "ao_shaping.tools.slm.panel.focus_probe"

#: 离焦组与远场组引用的相位文件 (相对 sweep 根)。
BEST_PHASE_NMAX4 = "zernike_order_nmax4/best_phase_nmax4.npy"
FIT_PHASE_TEMPLATE = "zernike_order_nmax{n}/phase_nmax{n}.npy"


class PlanError(RuntimeError):
    """计划自检没过 —— 带着具体原因退出, 而不是写出一份错的计划。"""


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def fmt_number(value: float) -> str:
    """数字 → 字符串 (去掉多余的尾零, 与上一轮计划里写的形式一致)。"""
    return f"{float(value):.10g}"


def parse_float_list(text: str) -> tuple[float, ...]:
    """把 ``"0 0 -1.5"`` / ``"0,0,-1.5"`` 解析成浮点元组。

    Args:
        text: 逗号或空白分隔的数串。

    Returns:
        浮点元组 (保持书写顺序)。

    Raises:
        ValueError: 串是空的, 或里面有不是数字的项。
    """
    parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
    if not parts:
        raise ValueError("点表不能为空")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"点表里有不是数字的项: {exc}") from exc


def parse_int_list(text: str) -> tuple[int, ...]:
    """把 ``"3,4,6"`` 解析成整数元组 (保持书写顺序)。"""
    parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
    if not parts:
        raise ValueError("阶数表不能为空")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"阶数表里有不是整数的项: {exc}") from exc


def arg_value(args: list[str], flag: str) -> str | None:
    """取 ``args`` 里 ``flag`` 后面那个值 (``None`` = 没这个参数)。"""
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return str(args[idx + 1])
    return None


def is_under(child: str, root: str) -> bool:
    """``child`` 是否在 ``root`` 之下 (相等也算)。

    纯词法比较 (不碰文件系统), 所以还没落盘的路径也能查 —— 这正是生成计划时
    要做的事: 计划里的路径**此刻**都还不存在。
    """
    c = PurePosixPath(str(child).replace("\\", "/"))
    r = PurePosixPath(str(root).replace("\\", "/"))
    return c == r or r in c.parents


def meta_args(meta: dict[str, str]) -> list[str]:
    """``{"k": "v"}`` → ``["--group-meta", "k=v", ...]`` (保持插入顺序)。"""
    out: list[str] = []
    for key, value in meta.items():
        out += ["--group-meta", f"{key}={value}"]
    return out


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class PlanConfig:
    """这一轮计划的全部可变量。"""

    groups_root: str = DEFAULT_GROUPS_ROOT
    sweep_root: str = DEFAULT_SWEEP_ROOT
    defocus_focal_mm: float = DEFAULT_DEFOCUS_FOCAL_MM

    total_epochs: int = DEFAULT_TOTAL_EPOCHS
    chunk_epochs: int = DEFAULT_CHUNK_EPOCHS
    epochs_per_chunk: int = DEFAULT_EPOCHS_PER_CHUNK
    time_budget_s: float = DEFAULT_TIME_BUDGET_S

    orders: tuple[int, ...] = DEFAULT_ORDERS
    dz_noise_mm: tuple[float, ...] = DEFAULT_DZ_NOISE_MM
    dz_scan_mm: tuple[float, ...] = DEFAULT_DZ_SCAN_MM
    dz_scan_model_mm: tuple[float, ...] = DEFAULT_DZ_SCAN_MODEL_MM

    hw_target_side_px: int = DEFAULT_HW_TARGET_SIDE_PX
    model_target_side_px: int = DEFAULT_MODEL_TARGET_SIDE_PX

    stage_a_exposure_ms: float = DEFAULT_STAGE_A_EXPOSURE_MS
    dz_hw_exposure_ms: float = DEFAULT_DZ_HW_EXPOSURE_MS
    dz_model_exposure_ms: float = DEFAULT_DZ_MODEL_EXPOSURE_MS
    farfield_exposure_ms: float = DEFAULT_FARFIELD_EXPOSURE_MS
    focus_exposure_ms: float = DEFAULT_FOCUS_EXPOSURE_MS
    focus_frames: int = DEFAULT_FOCUS_FRAMES
    focus_max_exposure_ms: float = DEFAULT_FOCUS_MAX_EXPOSURE_MS
    focus_steps: int = DEFAULT_FOCUS_STEPS
    focus_diopter_range: float = DEFAULT_FOCUS_DIOPTER_RANGE
    focus_refine: int = DEFAULT_FOCUS_REFINE
    frames: int = DEFAULT_FRAMES

    cam_type: str = DEFAULT_CAM_TYPE
    cam_id: int = DEFAULT_CAM_ID
    cam_size: int = DEFAULT_CAM_SIZE
    beam_radius_px: float = DEFAULT_BEAM_RADIUS_PX
    slm_number: int = DEFAULT_SLM_NUMBER
    gs_iters: int = DEFAULT_GS_ITERS
    spgd_epochs: int = DEFAULT_SPGD_EPOCHS
    lr: float = DEFAULT_LR
    delta: float = DEFAULT_DELTA
    max_energy_loss: float = DEFAULT_MAX_ENERGY_LOSS
    seed: int = DEFAULT_SEED

    plan_exposure_cap_ms: float = DEFAULT_PLAN_EXPOSURE_CAP_MS
    focus_exposure_cap_ms: float = DEFAULT_FOCUS_EXPOSURE_CAP_MS
    chirp_tilt_cancel: bool = True

    @property
    def chunks(self) -> int:
        """块数 = ``ceil(total_epochs / epochs_per_chunk)``。

        用的是**实际产出** (墙钟截断后的轮数), 不是请求值 —— 请求值是每块
        ``--chunk-epochs`` 的事, 两者由 ``chunk_epochs`` 与 ``epochs_per_chunk``
        分开表达。
        """
        if self.epochs_per_chunk <= 0:
            raise PlanError("每块实际产出必须为正")
        return math.ceil(self.total_epochs / self.epochs_per_chunk)

    @property
    def effective_epochs_per_chunk(self) -> int:
        """一块**真正**能贡献的轮数 = ``min(请求值, 实际产出)``。"""
        return min(self.chunk_epochs, self.epochs_per_chunk)

    @property
    def chunk_group_names(self) -> list[str]:
        """块组名 (``zernike_stage_a_1`` … ``zernike_stage_a_N``)。"""
        return [f"zernike_stage_a_{i}" for i in range(1, self.chunks + 1)]

    def stage_a_path(self, group: str) -> str:
        """某一块的交接文件路径 (``<sweep 根>/<组名>/stage_a.npz``)。"""
        return f"{self.sweep_root}/{group}/stage_a.npz"


# ---------------------------------------------------------------------------
# 各组
# ---------------------------------------------------------------------------
def focus_calibration_group(cfg: PlanConfig) -> dict[str, Any]:
    """焦面标定组。

    排在最前: 后面的每一组都用它标出来的 f_v, 报告的焦面那一节也从它的证据
    文件读。证据落在**本轮专属目录** —— 上一轮共用目录, 于是报告有可能读到
    另一次实验留下的证据。
    """
    debug_dir = f"{cfg.groups_root}/focus_calibration"
    return {
        "group": "focus_calibration",
        "variable": "f_v_mm",
        "meta": {
            "role": "focus_calibration",
            "phase": "hardware_gs",
            "exposure_ms": fmt_number(cfg.focus_exposure_ms),
            "debug_dir": debug_dir,
            "note": (
                "本轮的瞳面焦面标定: 证据落在本轮专属目录, 报告从这里读, "
                "不会读到另一次实验留下的证据"
            ),
        },
        "script": "-m",
        "args": [
            FOCUS_PROBE_MODULE,
            "--exposure-ms", fmt_number(cfg.focus_exposure_ms),
            "--frames", str(cfg.focus_frames),
            "--max-exposure-ms", fmt_number(cfg.focus_max_exposure_ms),
            "--steps", str(cfg.focus_steps),
            "--diopter-range", fmt_number(cfg.focus_diopter_range),
            "--refine", str(cfg.focus_refine),
            "--debug-dir", debug_dir,
        ],
        # 焦面标定要一路加曝光到 20 ms 扫阶梯 ⇒ 曝光上限必须比测量组高。
        "env": {"AO_DAHENG_MAX_EXPOSURE_MS": fmt_number(cfg.focus_exposure_cap_ms)},
    }


def stage_a_group(
    cfg: PlanConfig,
    index: int,
    name: str,
    previous: str | None,
) -> dict[str, Any]:
    """Stage A 的第 ``index`` 块。

    ``previous`` 是上一块的组名 (``None`` = 第一块)。第一块不带
    ``--stage-a-in``; 从第二块起读上一块的交接文件, 于是 250 轮被切成若干段
    跑, 而学习率仍按**全局** epoch 继续 (总地平线不变)。
    """
    meta = {
        "role": "zernike_stage_a",
        "chunk": f"{index}/{cfg.chunks}",
        "phase": "hardware_gs",
        "exposure_ms": fmt_number(cfg.stage_a_exposure_ms),
        "defocus_focal_mm": fmt_number(cfg.defocus_focal_mm),
        "chunk_epochs": str(cfg.chunk_epochs),
        "total_epochs": str(cfg.total_epochs),
    }
    args = [
        "--group", name,
        *meta_args(meta),
        "--cam-type", cfg.cam_type,
        "--cam-id", str(cfg.cam_id),
        "--exposure-time-ms", fmt_number(cfg.stage_a_exposure_ms),
        "--cam-size", str(cfg.cam_size),
        "--beam-radius-px", fmt_number(cfg.beam_radius_px),
        "--slm-number", str(cfg.slm_number),
        "--epochs", str(cfg.total_epochs),
        "--gs-iters", str(cfg.gs_iters),
        "--stage-a-only",
        "--chunk-epochs", str(cfg.chunk_epochs),
        "--time-budget-s", fmt_number(cfg.time_budget_s),
        "--lr", fmt_number(cfg.lr),
        "--delta", fmt_number(cfg.delta),
        "--max-energy-loss", fmt_number(cfg.max_energy_loss),
        "--seed", str(cfg.seed),
        "--defocus-focal-mm", fmt_number(cfg.defocus_focal_mm),
        "--save-images",
        "--out", cfg.sweep_root,
    ]
    if previous is not None:
        args += ["--stage-a-in", cfg.stage_a_path(previous)]
    return {
        "group": name,
        "variable": "chunk",
        "meta": meta,
        "script": STAGE_A_SCRIPT,
        "args": args,
    }


def zernike_order_group(
    cfg: PlanConfig,
    n_max: int,
    last_chunk: str,
) -> dict[str, Any]:
    """逐阶组: 把 Stage A 的最佳相位拟合到 ``n_max`` 阶泽尼克, 再跑 SPGD。

    读的是**最后一块**的 Stage A 产物 —— 中间块的交接文件只是续跑用的状态,
    最后一块才是完整 250 轮的结果。
    """
    name = f"zernike_order_nmax{n_max}"
    meta = {
        "role": "zernike_order",
        "phase": "hardware_gs",
        "exposure_ms": fmt_number(cfg.stage_a_exposure_ms),
        "defocus_focal_mm": fmt_number(cfg.defocus_focal_mm),
        "n_max": str(n_max),
    }
    return {
        "group": name,
        "variable": "n_max",
        "meta": meta,
        "script": STAGE_A_SCRIPT,
        "args": [
            "--group", name,
            *meta_args(meta),
            "--cam-type", cfg.cam_type,
            "--cam-id", str(cfg.cam_id),
            "--exposure-time-ms", fmt_number(cfg.stage_a_exposure_ms),
            "--cam-size", str(cfg.cam_size),
            "--beam-radius-px", fmt_number(cfg.beam_radius_px),
            "--slm-number", str(cfg.slm_number),
            "--spgd-epochs", str(cfg.spgd_epochs),
            "--lr", fmt_number(cfg.lr),
            "--delta", fmt_number(cfg.delta),
            "--max-energy-loss", fmt_number(cfg.max_energy_loss),
            "--seed", str(cfg.seed),
            "--defocus-focal-mm", fmt_number(cfg.defocus_focal_mm),
            "--save-images",
            "--out", cfg.sweep_root,
            "--stage-a-in", cfg.stage_a_path(last_chunk),
            "--orders", str(n_max),
        ],
    }


def _defocus_base_args(
    cfg: PlanConfig,
    group: str,
    exposure_ms: float,
    target_side_px: int,
    chirp_tilt_cancel: bool,
) -> list[str]:
    """离焦/远场组共用的前半段参数。"""
    args = [
        "--execute",
        "--group", group,
        "--out-root", cfg.groups_root,
        "--defocus-focal-mm", fmt_number(cfg.defocus_focal_mm),
        "--exposure-ms", fmt_number(exposure_ms),
        "--frames", str(cfg.frames),
        "--target-side-px", str(target_side_px),
    ]
    if chirp_tilt_cancel:
        args.append("--chirp-tilt-cancel")
    return args


def dz_group(
    cfg: PlanConfig,
    group: str,
    role: str,
    phase: str,
    exposure_ms: float,
    target_side_px: int,
    dz_points: tuple[float, ...],
    phase_file: str | None,
) -> dict[str, Any]:
    """离焦组 (硬件相位或模型相位)。

    ``phase_file`` 为 ``None`` = 模型相位对照组: 它**不带任何相位文件**, 由探针
    现场用已验证的模型生成 —— 这是「模型相位」与「硬件相位」两条曲线的分野。
    """
    meta = {
        "role": role,
        "phase": phase,
        "exposure_ms": fmt_number(exposure_ms),
        "defocus_focal_mm": fmt_number(cfg.defocus_focal_mm),
        "target_side_px": str(target_side_px),
        "chirp_tilt_cancel": "true" if cfg.chirp_tilt_cancel else "false",
    }
    if phase_file is not None:
        meta["phase_file"] = phase_file
    args = _defocus_base_args(
        cfg, group, exposure_ms, target_side_px, cfg.chirp_tilt_cancel
    )
    args += meta_args(meta)
    if phase_file is not None:
        args += ["--gs-phase", phase_file]
    args.append("--dz-mm")
    args += [fmt_number(v) for v in dz_points]
    return {
        "group": group,
        "variable": "dz_mm",
        "meta": meta,
        "script": DEFOCUS_SCRIPT,
        "args": args,
    }


def farfield_group(cfg: PlanConfig, n_max: int) -> dict[str, Any]:
    """远场补拍组: 用逐阶组落盘的泽尼克拟合相位, 在 Δz=0 拍一张远场。

    引用的相位是**本轮**逐阶组落盘的 ``phase_nmax{n}.npy`` —— 上一轮这里写的是
    旧根, 于是逐阶组失败后这一组静默用了上一轮遗留的相位而不报错。
    """
    name = f"zernike_farfield_nmax{n_max}"
    phase_file = f"{cfg.sweep_root}/{FIT_PHASE_TEMPLATE.format(n=n_max)}"
    # 远场补拍组是**单张** Δz=0 的远场, 不是一次扫描 ⇒ 不打开瞳面斜坡去倾斜
    # (那个开关是为扫描用的: 它在扫描区间里现场自标定「面板斜坡 → 相机位移」)。
    meta = {
        "role": "zernike_farfield",
        "n_max": str(n_max),
        "phase": "zernike_fit",
        "exposure_ms": fmt_number(cfg.farfield_exposure_ms),
        "defocus_focal_mm": fmt_number(cfg.defocus_focal_mm),
        "target_side_px": str(cfg.hw_target_side_px),
    }
    args = _defocus_base_args(
        cfg, name, cfg.farfield_exposure_ms, cfg.hw_target_side_px, False
    )
    args += meta_args(meta)
    args += [
        "--gs-phase", phase_file,
        "--dz-mm", "0",
    ]
    return {
        "group": name,
        "variable": "n_max",
        "meta": meta,
        "script": DEFOCUS_SCRIPT,
        "args": args,
    }


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def plan_comment(cfg: PlanConfig) -> str:
    """计划顶层的说明: 这一轮与上一轮差在哪, 为什么。"""
    return (
        f"第三轮 (2026-10-22) 的采集计划, 由 scripts/slm/build_hw_sweep_plan.py "
        f"生成。**新数据根**: 分组根 {cfg.groups_root} / 泽尼克扫描根 "
        f"{cfg.sweep_root} —— 上一轮与第二轮共用旧根, 于是第三轮的离焦/远场组把"
        f"第二轮那 11 个组的 measurements.json 覆盖了, 第二轮报告里的这些数字在"
        f"磁盘上已不可复现; 这一轮所有 --out-root / --out / --stage-a-in / 相位"
        f"文件路径都指向新根。"
        f"**块数 {cfg.chunks}**: 每块墙钟预算 {fmt_number(cfg.time_budget_s)} s, "
        f"实测一块只跑出 ~{cfg.epochs_per_chunk} 轮 (请求 {cfg.chunk_epochs} 轮), "
        f"{cfg.chunks} × {cfg.epochs_per_chunk} ≥ {cfg.total_epochs} 才够得到总"
        f"目标 —— 上一轮的 6 块 × 39 ≈ 234 轮够不到 {cfg.total_epochs} 轮。"
        f"**焦面标定排在最前**: 后面的每一组都用它标出来的 f_v "
        f"({fmt_number(cfg.defocus_focal_mm)} mm), 证据落在本轮专属目录。"
        f"**曝光上限由计划显式写死**: 测量组 "
        f"{fmt_number(cfg.plan_exposure_cap_ms)} ms (焦面标定组 "
        f"{fmt_number(cfg.focus_exposure_cap_ms)} ms, 它要扫到 "
        f"{fmt_number(cfg.focus_max_exposure_ms)} ms), 内置的加倍阶梯估计在暗"
        f"实验系统上会给出假上限。"
        f"**孔径预先登记**: 硬件相位 {cfg.hw_target_side_px} px (平场腰径 2·w0) "
        f"与模型相位 {cfg.model_target_side_px} px (模型方斑边长) 是两套相位各自"
        f"的定尺, 不可互换。"
    )


def build_plan(cfg: PlanConfig) -> dict[str, Any]:
    """按配置组装整份计划 (组序 = 焦面标定 → Stage A 块 → 逐阶 → 离焦 → 远场)。

    Raises:
        PlanError: 自检没过 (问题清单在异常消息里)。
    """
    groups: list[dict[str, Any]] = [focus_calibration_group(cfg)]

    chunk_names = cfg.chunk_group_names
    for index, name in enumerate(chunk_names, start=1):
        previous = chunk_names[index - 2] if index >= 2 else None
        groups.append(stage_a_group(cfg, index, name, previous))
    last_chunk = chunk_names[-1]

    for n_max in cfg.orders:
        groups.append(zernike_order_group(cfg, n_max, last_chunk))

    best_phase = f"{cfg.sweep_root}/{BEST_PHASE_NMAX4}"
    groups.append(
        dz_group(
            cfg,
            group="dz_noise_hw_gs",
            role="dz_noise",
            phase="hardware_gs",
            exposure_ms=cfg.dz_hw_exposure_ms,
            target_side_px=cfg.hw_target_side_px,
            dz_points=cfg.dz_noise_mm,
            phase_file=best_phase,
        )
    )
    groups.append(
        dz_group(
            cfg,
            group="dz_scan_hw_gs",
            role="dz_scan",
            phase="hardware_gs",
            exposure_ms=cfg.dz_hw_exposure_ms,
            target_side_px=cfg.hw_target_side_px,
            dz_points=cfg.dz_scan_mm,
            phase_file=best_phase,
        )
    )
    groups.append(
        dz_group(
            cfg,
            group="dz_scan_model_gs",
            role="dz_scan_control",
            phase="model_gs",
            exposure_ms=cfg.dz_model_exposure_ms,
            target_side_px=cfg.model_target_side_px,
            dz_points=cfg.dz_scan_model_mm,
            phase_file=None,
        )
    )

    for n_max in cfg.orders:
        groups.append(farfield_group(cfg, n_max))

    plan: dict[str, Any] = {
        "_comment": plan_comment(cfg),
        "root": cfg.groups_root,
        "groups": groups,
        "env": {"AO_DAHENG_MAX_EXPOSURE_MS": fmt_number(cfg.plan_exposure_cap_ms)},
    }
    problems = validate_plan(plan, cfg)
    if problems:
        raise PlanError("计划自检没过:\n  " + "\n  ".join(problems))
    return plan


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def check_unique_names(plan: dict[str, Any]) -> list[str]:
    """(a) 组名唯一。"""
    names = [str(g.get("group", "")) for g in plan.get("groups") or []]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        return [f"组名重复: {dupes}"]
    return []


def check_identity_fields(plan: dict[str, Any]) -> list[str]:
    """(e) 每组都带自变量与角色。

    空的自变量会让驱动的索引里「这一组扫的是什么」一栏是空的, 而这一栏正是
    事后查「哪个变量在动」的唯一线索。
    """
    problems: list[str] = []
    for group in plan.get("groups") or []:
        name = str(group.get("group", ""))
        if not str(group.get("variable", "") or "").strip():
            problems.append(f"组 {name} 没有自变量 (variable 为空)")
        meta = group.get("meta") or {}
        if not isinstance(meta, dict):
            problems.append(f"组 {name} 的 meta 不是对象")
            continue
        if not str(meta.get("role", "") or "").strip():
            problems.append(f"组 {name} 没有 meta.role")
    return problems


def check_chunk_chain(plan: dict[str, Any], cfg: PlanConfig) -> list[str]:
    """(b) 块链完整: 第 1 块无 ``--stage-a-in``, 第 N 块读第 N-1 块。

    而且**每一处** ``--stage-a-in`` 指到的组都必须在本计划里 —— 指向计划外的组
    就等于读上一轮的遗留文件, 而那不会报错。
    """
    problems: list[str] = []
    groups = plan.get("groups") or []
    names = {str(g.get("group", "")) for g in groups}
    by_name = {str(g.get("group", "")): g for g in groups}

    chunk_by_index: dict[int, str] = {}
    for name in names:
        match = re.fullmatch(r"zernike_stage_a_(\d+)", name)
        if match:
            chunk_by_index[int(match.group(1))] = name
    if sorted(chunk_by_index) != list(range(1, len(chunk_by_index) + 1)):
        problems.append(
            f"块编号不连续: {sorted(chunk_by_index)} "
            f"(应为 1..{len(chunk_by_index)})"
        )

    for index in sorted(chunk_by_index):
        name = chunk_by_index[index]
        args = [str(a) for a in (by_name[name].get("args") or [])]
        actual = arg_value(args, "--stage-a-in")
        if index == 1:
            if actual is not None:
                problems.append(f"第 1 块 {name} 不该带 --stage-a-in (却带了 {actual})")
            continue
        previous = chunk_by_index.get(index - 1)
        if previous is None:
            problems.append(f"第 {index} 块 {name} 的上一块不存在")
            continue
        expected = cfg.stage_a_path(previous)
        if actual != expected:
            problems.append(
                f"第 {index} 块 {name} 的 --stage-a-in 指向 {actual!r}, "
                f"应为 {expected!r}"
            )

    # 任何一组 (含逐阶组) 的 --stage-a-in 都必须指向计划内的组。
    for group in groups:
        name = str(group.get("group", ""))
        args = [str(a) for a in (group.get("args") or [])]
        actual = arg_value(args, "--stage-a-in")
        if actual is None:
            continue
        referenced = PurePosixPath(str(actual).replace("\\", "/")).parent.name
        if referenced not in names:
            problems.append(
                f"组 {name} 的 --stage-a-in 指向计划外的组 {referenced!r}"
            )

    # 逐阶组必须读**最后一块**: 中间块的交接文件只是续跑状态, 最后一块才是
    # 完整总轮数的结果。读错块 = 用半程的相位去拟合泽尼克。
    last_chunk = cfg.chunk_group_names[-1] if cfg.chunk_group_names else ""
    expected_last = cfg.stage_a_path(last_chunk)
    for group in groups:
        name = str(group.get("group", ""))
        if not name.startswith("zernike_order_nmax"):
            continue
        args = [str(a) for a in (group.get("args") or [])]
        actual = arg_value(args, "--stage-a-in")
        if actual != expected_last:
            problems.append(
                f"逐阶组 {name} 的 --stage-a-in 指向 {actual!r}, "
                f"应为最后一块的 {expected_last!r}"
            )
    return problems


def check_paths_under_roots(plan: dict[str, Any], cfg: PlanConfig) -> list[str]:
    """(c) 路径必须落在本轮的根下。

    这一条堵的是上一轮那个静默坑: 逐阶组失败后, 离焦组读了**旧根**里遗留的
    相位文件, 数据看起来齐全, 其实是上一轮的东西。
    """
    problems: list[str] = []
    for group in plan.get("groups") or []:
        name = str(group.get("group", ""))
        args = [str(a) for a in (group.get("args") or [])]
        meta = group.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        phase = arg_value(args, "--gs-phase")
        if phase is not None and not is_under(phase, cfg.sweep_root):
            problems.append(
                f"组 {name} 的 --gs-phase {phase!r} 不在本轮 sweep 根 "
                f"{cfg.sweep_root!r} 下"
            )
        phase_file = str(meta.get("phase_file", "") or "")
        if phase_file and not is_under(phase_file, cfg.sweep_root):
            problems.append(
                f"组 {name} 的 meta.phase_file {phase_file!r} 不在本轮 sweep 根 "
                f"{cfg.sweep_root!r} 下"
            )
        out = arg_value(args, "--out")
        if out is not None and not is_under(out, cfg.sweep_root):
            problems.append(
                f"组 {name} 的 --out {out!r} 不在本轮 sweep 根 {cfg.sweep_root!r} 下"
            )
        out_root = arg_value(args, "--out-root")
        if out_root is not None and not is_under(out_root, cfg.groups_root):
            problems.append(
                f"组 {name} 的 --out-root {out_root!r} 不在本轮分组根 "
                f"{cfg.groups_root!r} 下"
            )
        debug_dir = arg_value(args, "--debug-dir")
        if debug_dir is not None and not is_under(debug_dir, cfg.groups_root):
            problems.append(
                f"组 {name} 的 --debug-dir {debug_dir!r} 不在本轮分组根 "
                f"{cfg.groups_root!r} 下"
            )
    return problems


def check_epoch_coverage(plan: dict[str, Any], cfg: PlanConfig) -> list[str]:
    """(d) 块链能覆盖总轮数。"""
    effective = cfg.effective_epochs_per_chunk
    covered = cfg.chunks * effective
    if covered < cfg.total_epochs:
        return [
            f"块链覆盖不住总轮数: {cfg.chunks} 块 × {effective} 轮 = {covered} "
            f"< {cfg.total_epochs} 轮 (每块请求 {cfg.chunk_epochs} 轮, 墙钟内实际 "
            f"产出 {cfg.epochs_per_chunk} 轮)"
        ]
    return []


def validate_plan(plan: dict[str, Any], cfg: PlanConfig) -> list[str]:
    """跑完五项自检, 返回问题清单 (空 = 全过)。"""
    problems: list[str] = []
    problems += check_unique_names(plan)
    problems += check_identity_fields(plan)
    problems += check_chunk_chain(plan, cfg)
    problems += check_paths_under_roots(plan, cfg)
    problems += check_epoch_coverage(plan, cfg)
    return problems


def self_check_report(plan: dict[str, Any], cfg: PlanConfig) -> str:
    """自检的可读报告 (写到 stderr, 于是 ``--print-plan`` 的 stdout 仍是纯 JSON)。"""
    groups = plan.get("groups") or []
    names = [str(g.get("group", "")) for g in groups]
    chunk_by_index: dict[int, str] = {}
    for name in names:
        match = re.fullmatch(r"zernike_stage_a_(\d+)", name)
        if match:
            chunk_by_index[int(match.group(1))] = name
    phase_paths = 0
    for group in groups:
        args = [str(a) for a in (group.get("args") or [])]
        if arg_value(args, "--gs-phase") is not None:
            phase_paths += 1
        meta = group.get("meta") or {}
        if isinstance(meta, dict) and meta.get("phase_file"):
            phase_paths += 1
    effective = cfg.effective_epochs_per_chunk
    lines = [
        f"[自检] 组名唯一: 通过 ({len(names)} 组)",
        (
            f"[自检] 块链完整: 通过 ({cfg.chunks} 块, 编号 "
            f"{sorted(chunk_by_index)}, 第 N 块读第 N-1 块)"
        ),
        (
            f"[自检] 相位文件在本轮 sweep 根下: 通过 "
            f"({phase_paths} 处相位路径, 根 {cfg.sweep_root})"
        ),
        (
            f"[自检] 块链覆盖总轮数: 通过 ({cfg.chunks} 块 × {effective} 轮 = "
            f"{cfg.chunks * effective} ≥ {cfg.total_epochs})"
        ),
        "[自检] 每组带自变量与 meta.role: 通过",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--groups-root", default=DEFAULT_GROUPS_ROOT,
                        help="本轮分组根目录 (离焦/远场组与焦面标定证据)")
    parser.add_argument("--sweep-root", default=DEFAULT_SWEEP_ROOT,
                        help="本轮泽尼克扫描根目录 (Stage A / 逐阶组与相位文件)")
    parser.add_argument("--out", default=None,
                        help="计划 JSON 的落盘路径 (默认 <groups-root>/plan.json)")
    parser.add_argument("--print-plan", action="store_true",
                        help="把计划打到 stdout, 不落盘")

    parser.add_argument("--defocus-focal-mm", type=float,
                        default=DEFAULT_DEFOCUS_FOCAL_MM,
                        help="瞳面菲涅尔焦面矫正焦距 (mm), 出现在每一组里")
    parser.add_argument("--total-epochs", type=int, default=DEFAULT_TOTAL_EPOCHS,
                        help="Stage A 的总精化轮数 (学习率调度的总地平线)")
    parser.add_argument("--chunk-epochs", type=int, default=DEFAULT_CHUNK_EPOCHS,
                        help="每块**请求**的轮数 (墙钟预算只管得住实际产出)")
    parser.add_argument("--epochs-per-chunk", type=int,
                        default=DEFAULT_EPOCHS_PER_CHUNK,
                        help="一块在墙钟预算内**实际**跑出的轮数 (块数按它算)")
    parser.add_argument("--time-budget-s", type=float, default=DEFAULT_TIME_BUDGET_S,
                        help="Stage A 每块的墙钟预算 (s)")

    parser.add_argument("--orders", default=",".join(str(o) for o in DEFAULT_ORDERS),
                        help="逐阶组与远场补拍组扫的泽尼克截断阶数 (升序)")
    parser.add_argument("--dz-noise-mm",
                        default=" ".join(fmt_number(v) for v in DEFAULT_DZ_NOISE_MM),
                        help="噪声组的 Δz 点表 (同一 Δz 重复多次)")
    parser.add_argument("--dz-scan-mm",
                        default=" ".join(fmt_number(v) for v in DEFAULT_DZ_SCAN_MM),
                        help="硬件相位离焦扫描的 Δz 点表 (mm)")
    parser.add_argument("--dz-scan-model-mm",
                        default=" ".join(fmt_number(v) for v in DEFAULT_DZ_SCAN_MODEL_MM),
                        help="模型相位对照组的 Δz 点表 (mm)")

    parser.add_argument("--hw-target-side-px", type=int,
                        default=DEFAULT_HW_TARGET_SIDE_PX,
                        help="硬件相位组的目标框边长 (CCD px)")
    parser.add_argument("--model-target-side-px", type=int,
                        default=DEFAULT_MODEL_TARGET_SIDE_PX,
                        help="模型相位组的目标框边长 (CCD px)")

    parser.add_argument("--stage-a-exposure-ms", type=float,
                        default=DEFAULT_STAGE_A_EXPOSURE_MS)
    parser.add_argument("--dz-hw-exposure-ms", type=float,
                        default=DEFAULT_DZ_HW_EXPOSURE_MS)
    parser.add_argument("--dz-model-exposure-ms", type=float,
                        default=DEFAULT_DZ_MODEL_EXPOSURE_MS)
    parser.add_argument("--farfield-exposure-ms", type=float,
                        default=DEFAULT_FARFIELD_EXPOSURE_MS)
    parser.add_argument("--focus-exposure-ms", type=float,
                        default=DEFAULT_FOCUS_EXPOSURE_MS)
    parser.add_argument("--focus-frames", type=int, default=DEFAULT_FOCUS_FRAMES)
    parser.add_argument("--focus-max-exposure-ms", type=float,
                        default=DEFAULT_FOCUS_MAX_EXPOSURE_MS)
    parser.add_argument("--focus-steps", type=int, default=DEFAULT_FOCUS_STEPS)
    parser.add_argument("--focus-diopter-range", type=float,
                        default=DEFAULT_FOCUS_DIOPTER_RANGE)
    parser.add_argument("--focus-refine", type=int, default=DEFAULT_FOCUS_REFINE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)

    parser.add_argument("--cam-type", default=DEFAULT_CAM_TYPE)
    parser.add_argument("--cam-id", type=int, default=DEFAULT_CAM_ID)
    parser.add_argument("--cam-size", type=int, default=DEFAULT_CAM_SIZE)
    parser.add_argument("--beam-radius-px", type=float, default=DEFAULT_BEAM_RADIUS_PX)
    parser.add_argument("--slm-number", type=int, default=DEFAULT_SLM_NUMBER)
    parser.add_argument("--gs-iters", type=int, default=DEFAULT_GS_ITERS)
    parser.add_argument("--spgd-epochs", type=int, default=DEFAULT_SPGD_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--max-energy-loss", type=float, default=DEFAULT_MAX_ENERGY_LOSS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--plan-exposure-cap-ms", type=float,
                        default=DEFAULT_PLAN_EXPOSURE_CAP_MS,
                        help="计划级曝光上限 (AO_DAHENG_MAX_EXPOSURE_MS)")
    parser.add_argument("--focus-exposure-cap-ms", type=float,
                        default=DEFAULT_FOCUS_EXPOSURE_CAP_MS,
                        help="焦面标定组的曝光上限覆盖 (它要一路加到 20 ms)")
    parser.add_argument("--chirp-tilt-cancel", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="离焦/远场组是否打开瞳面斜坡去倾斜")
    return parser


def config_from_args(args: argparse.Namespace) -> PlanConfig:
    """命令行 → 配置。"""
    try:
        orders = parse_int_list(args.orders)
        dz_noise = parse_float_list(args.dz_noise_mm)
        dz_scan = parse_float_list(args.dz_scan_mm)
        dz_model = parse_float_list(args.dz_scan_model_mm)
    except ValueError as exc:
        raise PlanError(f"参数错误: {exc}") from exc
    return PlanConfig(
        groups_root=str(args.groups_root),
        sweep_root=str(args.sweep_root),
        defocus_focal_mm=float(args.defocus_focal_mm),
        total_epochs=int(args.total_epochs),
        chunk_epochs=int(args.chunk_epochs),
        epochs_per_chunk=int(args.epochs_per_chunk),
        time_budget_s=float(args.time_budget_s),
        orders=orders,
        dz_noise_mm=dz_noise,
        dz_scan_mm=dz_scan,
        dz_scan_model_mm=dz_model,
        hw_target_side_px=int(args.hw_target_side_px),
        model_target_side_px=int(args.model_target_side_px),
        stage_a_exposure_ms=float(args.stage_a_exposure_ms),
        dz_hw_exposure_ms=float(args.dz_hw_exposure_ms),
        dz_model_exposure_ms=float(args.dz_model_exposure_ms),
        farfield_exposure_ms=float(args.farfield_exposure_ms),
        focus_exposure_ms=float(args.focus_exposure_ms),
        focus_frames=int(args.focus_frames),
        focus_max_exposure_ms=float(args.focus_max_exposure_ms),
        focus_steps=int(args.focus_steps),
        focus_diopter_range=float(args.focus_diopter_range),
        focus_refine=int(args.focus_refine),
        frames=int(args.frames),
        cam_type=str(args.cam_type),
        cam_id=int(args.cam_id),
        cam_size=int(args.cam_size),
        beam_radius_px=float(args.beam_radius_px),
        slm_number=int(args.slm_number),
        gs_iters=int(args.gs_iters),
        spgd_epochs=int(args.spgd_epochs),
        lr=float(args.lr),
        delta=float(args.delta),
        max_energy_loss=float(args.max_energy_loss),
        seed=int(args.seed),
        plan_exposure_cap_ms=float(args.plan_exposure_cap_ms),
        focus_exposure_cap_ms=float(args.focus_exposure_cap_ms),
        chirp_tilt_cancel=bool(args.chirp_tilt_cancel),
    )


def main(argv: list[str] | None = None) -> int:
    """生成计划 (落盘或只打印)。

    Returns:
        0 = 计划已生成 (自检全过); 2 = 自检没过, 没有写出任何文件。
    """
    args = _build_parser().parse_args(argv)
    try:
        cfg = config_from_args(args)
        plan = build_plan(cfg)
    except PlanError as exc:
        logger.info(str(exc))
        return 2

    logger.info(self_check_report(plan, cfg))
    payload = json.dumps(plan, indent=2, ensure_ascii=False)
    if args.print_plan:
        # stdout 必须是纯 JSON (机器解析契约, 见 self_check_report docstring);
        # 自检报告走 logger -> stderr, 不能进 stdout。
        print(payload)
        return 0

    out = Path(args.out) if args.out else Path(cfg.groups_root) / "plan.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")
    print(f"计划已写入: {out} ({len(plan['groups'])} 组)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
