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

* 只 import 两个**零导入叶子** (:mod:`ao_shaping.utils.slm.bench_constants` 与
  :mod:`ao_shaping.utils.wavefront.matrix_utils`) —— 实测它们**不牵入任何
  ``ao_shaping.drivers.*``**;
* 波长常量 ``DEFAULT_WAVELENGTH_NM`` 用 ``ast`` 从 ``drivers/slm/factory.py``
  **静态读取** (不 import), 保持单一事实源又不触发驱动包的 import 期副作用;
* 收尾时**断言** ``ao_shaping.drivers`` 没出现在 ``sys.modules`` 里 —— 这条断言
  让「dry-run 不碰设备」成为可执行的保证而不是口头承诺, 违反就 exit 3。

因此 ``--dry-run`` 在**没有任何硬件的机器上**也必须 exit 0。

🔴 **不要按 M=(f+d)/f 重采样。** 有外部材料主张 CCD 挪到 f+d 时图样被放大 ``M``、
target 要按 ``M`` 重采样 —— **对 2f 光路是错的**, 错在漏掉了透镜。Collins/ABCD::

    T(f+d) · Lens(f) · T(f) = [[−d/f, f], [−1/f, 0]]      → A = −d/f, **B ≡ f**

核是 ``e^{−ikxξ/B}`` 而 ``B`` 恒等于 ``f`` ⇒ 核尺度是 ``λf`` 而非 ``λ(f+d)``, **没有
放大**。CCD 离焦只让图样**变模糊**: 单个倾斜平面波经透镜聚到 fθ 之后只是发散, 中心
不跑。本脚本因此不缩放 ROI、不按 d 重采样目标框 (目标框边长恒为 ``--target-side-px``),
瞳面二次相位的分母取 **λf²**。

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
* **零级必须 ``argmax`` 定位**, 绝不假设在画面中心 (2f 台架上画面中心 (1344,760)
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

产物写到 ``report/shape_trials/hw_gs_defocus_probe_<UTC 时间戳>/``: 每次测量一个
``.npz`` (原始帧 + 全部参数)、一份 ``measurements.csv``、一份 ``run_config.json``、
一张 ``hw_summary.png``。npz 里存的是**未经任何处理**的原始帧, 所以离线可以换度量
口径重新出图, 不必重连硬件。
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

# ---------------------------------------------------------------------------
# 光学常量: 全部来自**零导入叶子**, 导入它们不会牵入 ao_shaping.drivers.*
# (实测: sys.modules 里没有任何 ao_shaping.drivers 出现)。单一事实源仍是
# bench_constants / matrix_utils, 本文件不另立一份。
# ---------------------------------------------------------------------------
from ao_shaping.utils.slm.bench_constants import (  # noqa: E402
    BEAM_CENTER_PANEL,
    BEAM_RADIUS_PANEL,
    SLM_PANEL_H,
    SLM_PANEL_W,
    SLM_PITCH_M,
    TILT_SHIFT_SCALE,
)
from ao_shaping.utils.wavefront.matrix_utils import (  # noqa: E402
    focal_length_from_camera_pixel,
)

#: 2f 焦距标度是在哪个波长下测的 (bench_constants 的 TILT_SHIFT_SCALE 是 1064 nm 下测的)
SCALE_MEASURED_AT_NM = 1064.0
#: CCD 像元 (Daheng MER2-507-23GM, datasheet 常数, 见 drivers/AGENTS.md)
CAMERA_PIXEL_UM = 2.2

#: 瞳面二次相位的符号/前因子。**不是照抄公式**: 仿真脚本
#: ``experiment_gs_defocus_sweep_visuals.py`` 用方案A 的 |U_CCD|^2 数值对拍反解
#: (完整网格, 探针 Δz=0.1876mm):
#:   sign = -1 → 相对误差 4.46e-6 ; sign = +1 → 1.65e-3 (差 ~370 倍, 符号唯一)
#:   再对 α 做黄金分割 → alpha = 1.00016 (理论 1, 残差降到 2.24e-7)
#: 硬件侧沿用同一组值, 并可用 CLI 覆盖做复核。
CHIRP_SIGN = -1.0
CHIRP_ALPHA = 1.0001626284163687

#: 默认扫描网格 (mm)。**与仿真脚本的 ΔZ_SCAN_MM 逐点一致**, 两个脚本才能对拍。
DZ_SCAN_MM: tuple[float, ...] = tuple(float(v) for v in range(0, 55, 5))

#: 目标框边长 (CCD px)。默认值来自仿真的 43 远场 px × 2.2606 µm/px = 97.2 µm,
#: 换到 2.2 µm 的 CCD 像元上是 44.2 px —— 即仿真目标在硬件上的等效物理尺寸。
DEFAULT_TARGET_SIDE_PX = 44
DEFAULT_ROI_PX = 250

#: 每次读帧的超时保护 (s)。Santec 的 ``WaitImageV3`` 原生等待由 SDK 内部驱动,
#: **不受 Python 侧超时保护** (2026-09-08 实测可无限阻塞), 所以这里用看门狗线程
#: + 硬退出, 见 :func:`guarded`。
ACQUIRE_TIMEOUT_S = 60.0
OPEN_TIMEOUT_S = 180.0

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
        except BaseException as exc:  # noqa: BLE001 - 原样转交给主线程
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
def build_plan(args: argparse.Namespace, optics: Optics) -> dict[str, Any]:
    """完整实验计划: 扫哪些 Δz、每次施加什么相位、预计多久、需要哪些设备。"""
    dz_list = [float(v) for v in args.dz_mm]
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
                "action": (
                    "set_phase_overlay(瞳面二次相位) + display_and_average(GS 相位)"
                ),
                "frames": int(args.frames),
            }
        )
    # 时间预算: SLM 写入+稳定 ~0.5-1.5 s/次, CCD 取帧 ~0.1-0.3 s/帧。
    per_step = 1.5 + int(args.frames) * 0.3
    return {
        "devices": [
            {
                "role": "SLM",
                "driver": "ao_shaping.drivers.slm.santec.Santec (memory 模式, video_mode=0)",
                "slm_number": int(args.slm_number),
                "slm_wavelength_nm": int(args.wavelength_nm),
                "panel_shape_hw": list(optics.panel_shape),
                "fatal_if": "open() 在 DVI 模式会挂死 → 本脚本恒用 memory 模式",
            },
            {
                "role": "CCD",
                "driver": "ao_shaping.drivers.ccd.common.create_camera('daheng')",
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
        "overlay_api": (
            "Santec.set_phase_overlay(phase_rad) —— 弧度制, 在 create_phase_from_array "
            "内部、弧度→灰度之前相加, 矫正/LUT/平移照常作用; 结束后 set_phase_overlay(None)"
        ),
        "zero_order_halo": (
            "每次测量额外记整帧峰值 / 框外峰值 / 框外功率占比。真实瞳面是双分量 "
            "U=U_inc*[eta+(1-eta)e^{iφ}], 零级只过物理透镜, 在 f+d 处散成晕斑 "
            "(ρ≈d*D/(2f), I_halo ∝ f²/d²)。口径提示: Δz 小时零级峰仍在目标框内, "
            "框外峰值量不到它, 要看整帧峰值; Δz 大到散出框后框外峰值才代表污染"
        ),
        "no_magnification": (
            "不按 M=(f+d)/f 重采样: Collins/ABCD T(f+d)·Lens(f)·T(f) 的 B ≡ f ⇒ "
            "核尺度是 λf 而非 λ(f+d), CCD 离焦只使图样变模糊而不缩放。目标框边长恒为 "
            "--target-side-px, ROI 不随 d 改"
        ),
        "steps": steps,
        "estimate": {
            "per_step_s": per_step,
            "sweep_s": per_step * len(steps),
            "baseline_and_window_s": 12.0,
            "total_s": per_step * len(steps) + 12.0,
            "note": "不含 SLM/CCD open 与 close (各 5-60 s, SDK 而定)",
        },
        "outputs": (
            "report/shape_trials/hw_gs_defocus_probe_<UTC>/: 每次测量一个 npz "
            "(原始帧 + 参数) + measurements.csv + run_config.json + hw_summary.png; "
            "npz 存的是未处理原始帧, --replot 可离线重新出图"
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
    add("零级晕斑 (硬件侧最该量的东西)")
    add(f"  {plan['zero_order_halo']}")
    add("")
    add("离焦施加方式")
    add(f"  {plan['overlay_api']}")
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
    out_dir = (ROOT / "report" / "shape_trials" / f"hw_gs_defocus_probe_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"测量目录: {out_dir}")

    gs_panel = load_gs_phase_panel(args, optics)
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

    with guarded("SLM 打开", lambda: open_slm(params), timeout_s=OPEN_TIMEOUT_S) as slm:
        with guarded("SLM open()", lambda: slm.open(), timeout_s=OPEN_TIMEOUT_S):
            sync_device_wavelength(params, slm)
            with create_camera(params.cam_type, params.cam_id,
                               exposure_time_ms=params.exposure_time_ms) as cam:
                with preserve_slm_display(slm):
                    # -- 槽位轮换: 排除运行前正在显示的那个, 之后每次随机挑 ----------
                    try:
                        protected = slm.get_displayed_memory_number()
                    except SantecError as exc:
                        if exc.code != 1:
                            raise
                        protected = None  # set_grayscale 模式下没有槽位, 报错码 1 正常
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
                    flat = np.zeros(optics.panel_shape, dtype=np.float64)
                    full = capture(slm, cam, flat, next_slot())
                    cx_full, cy_full = zero_order_center(full, refine=False, despike_k=3)
                    print(f"零级 (全幅 argmax): ({cx_full}, {cy_full})")
                    cam.reset_window(center=(cx_full, cy_full),
                                     size=(int(args.roi_px), int(args.roi_px)))
                    roi_flat = capture(slm, cam, flat, next_slot())
                    cx, cy = zero_order_center(roi_flat, refine=False, despike_k=3)
                    # ⚠️ 冻结: 之后所有测量复用这一个 center 与框, 不逐帧重定位。
                    target = square_mask(roi_flat.shape, cy, cx, int(args.target_side_px))
                    print(f"零级 (ROI 内 argmax, 已冻结): ({cx}, {cy}), "
                          f"target_side={args.target_side_px} px")

                    run_config = {
                        "timestamp_utc": stamp,
                        "plan": plan,
                        "gs_phase_source": plan["gs_phase_source"],
                        "exposure_ms_requested": float(args.exposure_ms),
                        "exposure_ceiling_ms": float(ceiling),
                        "exposure_ms_working": actual,
                        "frames_per_measurement": int(args.frames),
                        "zero_order_full_frame_xy": [int(cx_full), int(cy_full)],
                        "zero_order_roi_xy_frozen": [int(cx), int(cy)],
                        "roi_px": int(args.roi_px),
                        "target_side_px": int(args.target_side_px),
                        "slm_number": int(args.slm_number),
                        "cam_id": int(args.cam_id),
                        "cam_type": args.cam_type,
                        "protected_memory_slot": protected,
                    }
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
                            **{k: step[k] for k in
                               ("dz_mm", "dz_units", "chirp_sign", "chirp_alpha",
                                "chirp_edge_rad", "chirp_edge_waves", "chirp_corner_rad")},
                            "exposure_ms_readback": float(cam.get_exposure_time_ms()),
                            "frame_mean_in_target": float(np.mean(frame[target])),
                            "cv": cv,
                            "pib": pib,
                            "score": float(composite_from_pib_cv(pib, cv)),
                            **halo_metrics(frame, target),
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
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        return row

                    # -- 基线行 (Δz = 0, GS 相位, 无叠加) ---------------------------
                    baseline = {"step": -1, "dz_mm": 0.0, "dz_units": 0.0,
                                "chirp_sign": 0.0, "chirp_alpha": 0.0,
                                "chirp_edge_rad": 0.0, "chirp_edge_waves": 0.0,
                                "chirp_corner_rad": 0.0}
                    slm.set_phase_overlay(None)
                    record(baseline, capture(slm, cam, gs_panel, next_slot()))

                    # -- 扫描 --------------------------------------------------------
                    try:
                        for step in plan["steps"]:
                            chirp = panel_pupil_chirp(
                                step["dz_mm"] * 1e-3, optics,
                                sign=args.chirp_sign, alpha=args.chirp_alpha,
                            )
                            slm.set_phase_overlay(chirp)
                            record(step, capture(slm, cam, gs_panel, next_slot()))
                    finally:
                        # 叠加层是驱动的持久状态, 必须清掉 —— 否则之后的调用方会在
                        # 毫不知情的情况下一直带着这张离焦相位图。
                        slm.set_phase_overlay(None)

    write_measurements(out_dir, rows)
    plot_summary(out_dir, rows)
    print(f"\n产物目录: {out_dir}")
    return out_dir


def write_measurements(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    """把测量行写成 CSV + JSON (离线重新出图的输入)。"""
    if not rows:
        return
    with (out_dir / "measurements.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
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


def run_replot(args: argparse.Namespace) -> None:
    """离线重新出图: 只读测量目录里的 npz/CSV, **不碰任何设备**。"""
    src = Path(args.replot)
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
    p.add_argument("--slm-number", type=int, default=1)
    p.add_argument("--cam-id", type=int, default=0)
    p.add_argument("--cam-type", default="daheng")
    p.add_argument("--wavelength-nm", type=float, default=0.0,
                   help="0 = 静态读取 drivers/slm/factory.DEFAULT_WAVELENGTH_NM")
    p.add_argument("--exposure-ms", type=float, default=0.12)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--roi-px", type=int, default=DEFAULT_ROI_PX)
    p.add_argument("--target-side-px", type=int, default=DEFAULT_TARGET_SIDE_PX)
    p.add_argument("--settle-wait-s", type=float, default=0.5)
    p.add_argument("--settle-max-wait-s", type=float, default=6.0)
    p.add_argument("--settle-max-discard", type=int, default=40)
    p.add_argument("--seed", type=int, default=0, help="槽位随机选择的种子")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.replot is not None:
        run_replot(args)
        return

    optics = build_optics(args)  # 这里顺带把 --wavelength-nm 0 解析成设备常量
    plan = build_plan(args, optics)

    if args.execute:
        if args.dry_run:
            print("给了 --dry-run 和 --execute, 按 dry-run 处理", file=sys.stderr)
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