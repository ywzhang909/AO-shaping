"""对照实验: 光路复用判据能不能把「同一条光路」与「光路变了」分开?

Question
--------
``optical_probe.compare_flat_frames`` 用「亚像素对心 + 归一化互相关 (NCC) + 二阶矩
三轴」比平场斑形状; 生产判决 (``assess_reuse``) = NCC >= ``DEFAULT_MIN_FLAT_SIMILARITY``、
零点漂移 <= ``DEFAULT_MAX_ZERO_DRIFT_PX`` 硬上限、二阶矩三轴 ``within_tolerance``。
离线在**今天实测的真斑**上做已知量变换 (2026-11-11) 得到两个相反方向的结论:

* NCC 对**纯平移**与**只改亮度**完全免疫 —— 40 px 平移 1.0000, 曝光 x0.25..x3 全
  1.0000。这正是想要的: 相机/载物台挪一点点、曝光改一点, 都不该算「换了光路」;
* NCC 对**形变**也几乎免疫 —— 各向异性 20% → 0.9874, 旋转 12° → 0.9963, 等比放大
  30% (离焦代理) → 0.976, 全部躺在 0.90 阈值之上。也就是**只判 NCC 会漏判**。

本脚本上真设备把这件事量出来: 一场会话 (SLM #1 只开一次), 以平场斑为参照, 逐条件
下发相位或改曝光, 每条件都问生产判据一句 —— 该说「同一条」还是「变了」?

期望标签 (三值; 2026-11-11 21:28 实验之后按实测证据改标)
--------------
必须放行 (同一条光路, ``expect="same"``)
    * ``flat-repeat``: 同会话重复平场帧 (含真实抖动与零点漂移);
    * ``exposure-*``: 只改曝光, 未过曝 (亮度变、斑形不变);
    * ``tilt-2.0/3.7/7.4px``: 斜栅把整斑**刚性**平移几像素 (斑形不变、只有位置变)。

必须拒绝 (光路变了, ``expect="diff"``)
    * ``tilt-14.8px``: 该档 ramp 实测**不是刚性平移** —— 斑被真实拉伸 (相对参照
      sy=7.06 px, 该档 sy=9.90、sx 7.31→6.93、椭圆度 1.387、主轴 89.41°、
      NCC 0.894)。刺激源本身就是形变样本, 不是平移样本, 期望由 same 改为 diff;
    * ``defocus-*`` (+1000 mm 除外): 离焦, 斑整体变大/变小 (正负两支都测, 本仓不
      假设相位符号);
    * ``astig0-2.0rad``: ``(n,m)=(2,2)`` 像散 (2.0 rad RMS), 斑被非等比拉伸;
    * ``astig45-2.0rad``: ``(n,m)=(2,-2)`` 像散 (2.0 rad RMS), 拉伸方向转 45° ——
      「旋转」与「非等比」的合体样本;
    * ``coma-*`` / ``spherical-*``: 彗差/球差, 斑形畸变 (尾巴/环)。

低于探针可测下限 (``expect="below_floor"``, 不参与对错判定)
    * ``defocus-+1000mm``、``astig0-0.5rad``、``astig45-0.5rad``、``astig0-1.0rad``、
      ``astig45-1.0rad``: 刺激构造正确, 但幅度低于二阶矩判据的物理下限 (见下节),
      任何一种判决都接受 (``agrees=None``, 不进 mismatches); 混淆表单独计数,
      ``consistent = agrees + below_floor``。

低于下限的物理依据 (为什么这 5 条不算判错)
------------------------------------------
刺激构造本身正确 (Zernike 用 Noll 单位 RMS 归一、幅值以 rad RMS 计、
``radius=BEAM_RADIUS_PANEL`` 对齐面板光束半径; 离焦用 Fresnel 透镜相位), 但幅度落在
二阶矩判据的物理下限之下, 生产判据「看不见」, 故任何一种判决都不算错:

* 判据下限 (生产常量): 各向异性 ``aspect_mismatch`` 与等比 ``size_ratio`` 的容差都是
  1.12 (``DEFAULT_MAX_ASPECT_MISMATCH`` / ``DEFAULT_MAX_SIZE_RATIO``); 各向同性模糊
  下限 σ_blur ≈ 4.5 px。低于这些, 二阶矩三轴全部 ``within_tolerance`` → 判「同一条」。
* 像散: 离线估算 1 rad RMS 像散的波前曲率差仅相当于 ~26 m 焦距量级 (极弱); 实测
  0.5 / 1.0 rad 两档 ``aspect_mismatch`` 中位 ≈ 1.01-1.02、``size_ratio`` ≈ 0.95-1.01,
  全压在 1.12 之下, 要到 2.0 rad 各向异性才越过下限 (故 ``astig*-2.0rad`` 仍判 diff)。
* 离焦: ``defocus-+1000mm`` 实测 ``size_ratio`` 中位 ≈ 1.02 (光斑只大约 2%), 低于
  1.12 的 size 判据; ``defocus-+2000mm`` 才到 ≈ 1.19 越过下限。故离焦可测下限落在
  +1000 与 +2000 mm 之间, ``defocus-+1000mm`` 归 below_floor。

为什么斜栅可以当「平移」的代理
------------------------------
相机搬不动, 能在一条光路上把斑整体挪走的只有倾斜相位。``ramp_panel(P)`` 把 0 级
能量搬到 ``TILT_SHIFT_SCALE / P`` 相机像素之外 (2026-09-30 实测常数), 斑形不变、
只有位置变。位移越小残余 0 级与位移级重叠越多、越接近纯平移, 所以这里只用 2-15 px
的小位移。**这不是**纯几何平移: 它还换了衍射级, 所以它是最严格意义上的「同一条光路
上换了相位」, 若判据把它判成「变了」, 那对真实相机微移是保守方向, 可接受。

判据全部走生产代码
------------------
相似度/形状描述量只来自 ``compare_flat_frames`` (``FlatComparison`` 及其
``shape_comparison``); 脚本不另写第二份实现。这里只做三件事: 汇总分布、把生产常数
套成一句判决、把「期望 vs 判决」摆成混淆表。

Devices
-------
只开 **一次** SLM #1 + 相机 #0, 全程一个进程 (背对背开关会让驱动板假死, 只能手动
重启)。开场做相位自检 (``current_grayscale == 0``); 结束时下发平场相位并关闭, 在
报告里写明占用与释放。

Outputs
-------
``data/debug/slm_transfer/<YYYYMMDD_HHMMSS>_optical_reuse_discrim/``:

* ``frames.npz``: 每个条件的单帧块 (键 = 条件标签);
* ``plan.json``: 条件表 (标签/期望/曝光/相位说明);
* ``analysis.json``: 每条件的相似度分布、零点漂移、形状描述量、生产判决与混淆表。

复算
----
``--analyse <run_dir>`` 只读 ``frames.npz`` 重跑分析, **不开设备** —— 调阈值或改
形状判据之后, 不必再实验。

Delete when
-----------
阈值与形状判据固化进 ``optical_probe`` (含常量注释里的实测依据), 且后续会话不再需要
重新量这四个分布; 本脚本与它的 run 目录一起删。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

#: 仓库根 (从 ``scripts/slm/<this file>`` 上溯两级)。
REPO = Path(__file__).resolve().parents[2]

#: SLM 面板尺寸 ``(height, width)``; 与 ``measure_probe`` 下发的平场相位一致。
PANEL_SHAPE = (1200, 1920)

#: 曝光扫描 (ms)。实测 0.02 ms 下平场斑峰值 ~73 counts、背景 0.3, 饱和约在 0.07 ms,
#: 所以可用区间取 0.005-0.05, 每一档都该判「同一条」。
EXPOSURE_SWEEP_MS: tuple[float, ...] = (0.005, 0.010, 0.020, 0.035, 0.050)

#: 期望的整斑位移 (相机 px); 周期由 ``P = TILT_SHIFT_SCALE / shift`` 反推。
TILT_SHIFTS_PX: tuple[float, ...] = (2.0, 3.7, 7.4, 14.8)

#: 离焦焦距 (mm); 正负两支都测 (相位符号约定不在这里假设)。
DEFOCUS_MM: tuple[float, ...] = (4000.0, 2000.0, 1000.0, -1000.0, -2000.0)

#: Zernike 幅值 (rad, 单位 RMS Noll 归一)。
ZERNIKE_AMPLITUDES: tuple[float, ...] = (0.5, 1.0, 2.0)

#: 主样本条件标签 (参照帧来源)。
REFERENCE_LABEL = "flat-repeat"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行 (默认离线: 只打印条件表, 不开设备)。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="真的打开 SLM #1 + 相机 #0 (不加只打印计划)",
    )
    parser.add_argument("--camera", default="daheng", help="相机后端名")
    parser.add_argument("--cam-id", type=int, default=0, help="相机 id")
    parser.add_argument("--slm-number", type=int, default=1, help="SLM 编号")
    parser.add_argument(
        "--exposure-ms",
        type=float,
        default=0.02,
        help="基准曝光 (ms), 与 measure_probe 的探针帧一致",
    )
    parser.add_argument("--frames", type=int, default=5, help="每条件抓几张单帧")
    parser.add_argument("--discard", type=int, default=1, help="切换条件后丢弃几张")
    parser.add_argument(
        "--settle-s", type=float, default=0.4, help="下发相位后等多久再抓 (s)"
    )
    parser.add_argument(
        "--wait-until-idle",
        type=float,
        default=0.0,
        help="开场等台面空闲的预算 (s); 0 = 只报告不等",
    )
    parser.add_argument(
        "--analyse",
        default=None,
        help="只复算: 指向已有 run 目录 (含 frames.npz), 不开设备",
    )
    parser.add_argument("--out-dir", default=None, help="输出目录 (默认时间戳)")
    return parser.parse_args(argv)


def default_out_dir() -> Path:
    """本次运行的输出目录 (时间戳命名, 与 slm_transfer 其它 run 同构)。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO / "data" / "debug" / "slm_transfer" / f"{stamp}_optical_reuse_discrim"


def report_occupancy(phase: str, devices: str) -> None:
    """按 ``docs/dev/hardware_discipline.md`` 报告设备占用 (开始/结束各一次)。"""
    logger.info(f"[设备占用·{phase}] {devices}")


def precheck_bench(budget_s: float) -> dict:
    """只读检查台面: 谁在占用, 需要的话等它空闲 (不打开任何设备)。"""
    from ao_shaping.tools.kernels.device_mutex import (
        find_device_using_processes,
        wait_until_idle,
    )

    record: dict[str, Any] = {
        "budget_s": float(budget_s),
        "occupiers": [],
        "wait": None,
    }
    hits = find_device_using_processes(own_paths=[__file__])
    record["occupiers"] = [{"pid": proc.pid, "cmdline": proc.cmdline} for proc in hits]
    if hits:
        logger.warning(f"台面占用者 {len(hits)} 个: {record['occupiers']}")
    if budget_s > 0:
        verdict = wait_until_idle(budget_s, own_paths=[__file__])
        record["wait"] = {
            "status": verdict.status,
            "waited_s": float(verdict.waited_s),
            "reason": verdict.reason,
        }
        logger.info(
            f"等空闲: status={verdict.status} waited={verdict.waited_s:.1f}s "
            f"{verdict.reason}"
        )
    return record


def slm_self_check(slm: object) -> dict:
    """开场相位自检: ``get_current_grayscale()`` 必须是 0 (平场/关)。

    按本仓约定, ``get_displayed_memory_number()`` 抛 ``SantecError(code=1)`` 在
    grayscale 模式下属正常 (没有 memory slot); ``get_displayed_phase()`` 只是驱动侧
    写缓存, 不是设备回读。
    """
    from ao_shaping.drivers.slm.santec import SantecError

    record: dict[str, Any] = {}
    get_gray = getattr(slm, "get_current_grayscale", None)
    if get_gray is None:
        record["grayscale"] = None
    else:
        record["grayscale"] = float(get_gray())
        if record["grayscale"] != 0.0:
            raise RuntimeError(
                f"SLM 打开后灰度不是 0 (读回 {record['grayscale']}) —— 先别下发相位"
            )
    get_slot = getattr(slm, "get_displayed_memory_number", None)
    if get_slot is not None:
        try:
            record["memory_slot"] = get_slot()
        except SantecError as exc:
            if exc.code != 1:
                raise
            record["memory_slot"] = None
    logger.info(f"SLM 相位自检: {record}")
    return record


def grab(cam: object, count: int) -> np.ndarray:
    """连抓 ``count`` 张**单帧** (不做平均), 返回 ``(count, H, W)``。

    单帧是形状判据最严格的一侧: 平均会抹掉噪声, 而真实探针帧就是单帧。
    """
    frames = [cam.get_numpy_image(n_sample=1, skip_first=False) for _ in range(count)]
    return np.stack([np.asarray(frame) for frame in frames])


def expected_by_label() -> dict[str, str]:
    """每个条件标签的期望 (三值: ``same`` / ``diff`` / ``below_floor``)。

    ``build_conditions`` (计划/下发) 与 ``--analyse`` (离线复算) 共用这一张表: 复算时
    用它覆盖 run 目录 ``plan.json`` 里的旧期望, 不会退回改标前的标签。``below_floor``
    的判定依据见模块 docstring「低于下限的物理依据」。
    """
    exp: dict[str, str] = {REFERENCE_LABEL: "same"}
    for ms in EXPOSURE_SWEEP_MS:
        exp[f"exposure-{ms * 1000:.0f}us"] = "same"
    for shift in TILT_SHIFTS_PX:
        # 14.8 px 这档 ramp 实测把斑真实拉伸 (非刚性平移), 故由 same 改判 diff。
        exp[f"tilt-{shift:.1f}px"] = "diff" if shift >= 14.8 else "same"
    for focal in DEFOCUS_MM:
        # +1000 mm 光斑只大约 2%, 低于 1.12 的 size 判据 → below_floor。
        exp[f"defocus-{focal:+.0f}mm"] = "below_floor" if focal == 1000.0 else "diff"
    for amp in ZERNIKE_AMPLITUDES:
        for tag in ("astig0", "astig45"):
            # 0.5 / 1.0 rad 各向异性压在 1.12 之下 → below_floor; 2.0 rad 越过下限。
            exp[f"{tag}-{amp:.1f}rad"] = "below_floor" if amp < 2.0 else "diff"
    exp["coma-1.0rad"] = "diff"
    exp["spherical-1.0rad"] = "diff"
    return exp


def build_conditions() -> list[dict[str, Any]]:
    """条件表: 每项一个 ``(标签, 期望, 相位, 曝光, 说明)``。

    期望统一取自 :func:`expected_by_label`, 保证计划与离线复算同一套标签。
    """
    from ao_shaping.tools.kernels.bench_kernels import (
        BEAM_CENTER_PANEL,
        BEAM_RADIUS_PANEL,
        TILT_SHIFT_SCALE,
        fresnel_panel,
        ramp_panel,
        zernike_panel,
    )

    expects = expected_by_label()
    flat = np.zeros(PANEL_SHAPE, dtype=np.float64)
    conditions: list[dict[str, Any]] = [
        {
            "label": REFERENCE_LABEL,
            "expect": expects[REFERENCE_LABEL],
            "phase": flat,
            "exposure_ms": None,
            "note": "重复平场帧: 同会话正样本分布 (参照帧由这一块求均值)",
        }
    ]
    for ms in EXPOSURE_SWEEP_MS:
        label = f"exposure-{ms * 1000:.0f}us"
        conditions.append(
            {
                "label": label,
                "expect": expects[label],
                "phase": flat,
                "exposure_ms": float(ms),
                "note": f"只改曝光到 {ms:g} ms (亮度变、斑形不变)",
            }
        )
    for shift in TILT_SHIFTS_PX:
        period = int(round(TILT_SHIFT_SCALE / shift))
        label = f"tilt-{shift:.1f}px"
        conditions.append(
            {
                "label": label,
                "expect": expects[label],
                "phase": ramp_panel(period, axis=1),
                "exposure_ms": None,
                "note": (
                    f"斜栅 period={period} px → 预测整斑位移 "
                    f"{TILT_SHIFT_SCALE / period:.2f} px"
                ),
            }
        )
    for focal in DEFOCUS_MM:
        label = f"defocus-{focal:+.0f}mm"
        conditions.append(
            {
                "label": label,
                "expect": expects[label],
                "phase": fresnel_panel(focal),
                "exposure_ms": None,
                "note": f"离焦 f={focal:+.0f} mm (斑整体变大/变小)",
            }
        )
    for amp in ZERNIKE_AMPLITUDES:
        for order, tag, why in (
            ((2, 2), "astig0", "0° 像散 x²-y²: 非等比拉伸"),
            ((2, -2), "astig45", "45° 像散 xy: 拉伸方向转了 45°"),
        ):
            label = f"{tag}-{amp:.1f}rad"
            conditions.append(
                {
                    "label": label,
                    "expect": expects[label],
                    "phase": zernike_panel(
                        {order: amp},
                        radius=BEAM_RADIUS_PANEL,
                        pupil_center=BEAM_CENTER_PANEL,
                    ),
                    "exposure_ms": None,
                    "note": f"{why}, 幅值 {amp:.1f} rad RMS",
                }
            )
    for order, tag, why in (
        ((3, 1), "coma", "彗差 (3,1): 斑拖尾"),
        ((4, 0), "spherical", "球差 (4,0): 斑外包环"),
    ):
        label = f"{tag}-1.0rad"
        conditions.append(
            {
                "label": label,
                "expect": expects[label],
                "phase": zernike_panel(
                    {order: 1.0},
                    radius=BEAM_RADIUS_PANEL,
                    pupil_center=BEAM_CENTER_PANEL,
                ),
                "exposure_ms": None,
                "note": f"{why}, 幅值 1.0 rad RMS",
            }
        )
    return conditions


def describe_plan(conditions: list[dict[str, Any]], exposure_ms: float) -> str:
    """条件表的可读摘要 (离线模式打印这一行集合)。"""
    lines = [f"基准曝光 {exposure_ms:g} ms, 共 {len(conditions)} 个条件:"]
    for cond in conditions:
        exp = cond["exposure_ms"]
        exposure = "基准" if exp is None else f"{exp:g} ms"
        lines.append(
            f"  {cond['label']:<20s} 期望={cond['expect']:<4s} 曝光={exposure:<9s}"
            f" {cond['note']}"
        )
    return "\n".join(lines)


def run_device(args: argparse.Namespace, out: Path) -> dict:
    """单次开设备: 逐条件下发相位/改曝光抓帧, 复位平场后关闭。"""
    from ao_shaping.drivers.ccd.common import create_camera
    from ao_shaping.drivers.slm.santec import Santec
    from ao_shaping.tools.slm.transfer.hardware import (
        opened_device,
        prepare_camera_sdk,
    )

    conditions = build_conditions()
    prepare_camera_sdk(args.camera)
    cam = create_camera(args.camera, args.cam_id, exposure_time_ms=args.exposure_ms)
    base_exposure = float(args.exposure_ms)
    result: dict[str, Any] = {
        "exposure_ms": base_exposure,
        "frames_per_condition": int(args.frames),
        "slm_number": int(args.slm_number),
        "steps": [],
    }
    blocks: dict[str, np.ndarray] = {}
    closed_normally = False

    report_occupancy(
        "开始",
        f"SLM #{args.slm_number} + 相机 {args.camera}#{args.cam_id} (激光常开)",
    )
    try:
        with opened_device(cam) as cam_handle:
            with opened_device(
                Santec(slm_number=args.slm_number, wavelength=1064, video_mode=0)
            ) as slm:
                result["slm_check"] = slm_self_check(slm)
                result["slm_serial"] = str(getattr(slm, "serial_number", None) or "")
                slm.set_phase_overlay(None)
                for cond in conditions:
                    exposure = cond["exposure_ms"]
                    effective = cam_handle.set_exposure_ms(
                        base_exposure if exposure is None else float(exposure)
                    )
                    slm.display_data(slm.create_phase_from_array(cond["phase"]))
                    time.sleep(float(args.settle_s))
                    if args.discard > 0:
                        grab(cam_handle, int(args.discard))
                    blocks[cond["label"]] = grab(cam_handle, int(args.frames))
                    result["steps"].append(
                        f"{cond['label']}: 曝光 {effective:g} ms, 抓 {args.frames} 张"
                    )
                    logger.info(
                        f"[{cond['label']}] 曝光 {effective:g} ms 抓 {args.frames} 张"
                    )
                # 复位: 平场相位 + 基准曝光 (与 measure_probe / 采集段 finally 一致)
                cam_handle.set_exposure_ms(base_exposure)
                slm.display_data(slm.create_phase_from_array(np.zeros(PANEL_SHAPE)))
        closed_normally = True
    finally:
        report_occupancy(
            "结束",
            f"SLM #{args.slm_number} + 相机 {args.camera}#{args.cam_id} 已释放"
            f" (正常关闭={closed_normally})",
        )

    result["closed_normally"] = closed_normally
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "frames.npz", **blocks)
    (out / "plan.json").write_text(
        json.dumps(
            [
                {key: value for key, value in cond.items() if key != "phase"}
                for cond in conditions
            ],
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result["conditions"] = [
        {key: value for key, value in cond.items() if key != "phase"}
        for cond in conditions
    ]
    return result


def _saturation(block: np.ndarray) -> int:
    """块里达到该 dtype 满量程的像素数 (整数帧才有意义, 浮点返回 0)。"""
    dtype = np.asarray(block).dtype
    if not np.issubdtype(dtype, np.integer):
        return 0
    full = np.iinfo(dtype).max
    return int((np.asarray(block) >= full).sum())


def analyse_blocks(
    blocks: dict[str, np.ndarray],
    conditions: list[dict[str, Any]],
    *,
    min_similarity: float,
    max_drift_px: float,
) -> dict[str, Any]:
    """对 ``frames.npz`` 里的每个条件块算生产判据, 并摆成混淆表。

    Args:
        blocks: 条件标签 → ``(n, H, W)`` 单帧块。
        conditions: 条件表 (要 ``expect`` / ``note`` / ``exposure_ms``)。
        min_similarity: 生产相似度阈值。
        max_drift_px: 生产零点漂移硬上限 (px)。

    Returns:
        可 JSON 序列化的分析结果 (含每条件行与混淆表)。
    """
    from ao_shaping.tools.slm.transfer.optical_probe import compare_flat_frames

    if REFERENCE_LABEL not in blocks:
        raise KeyError(f"缺少参照块 {REFERENCE_LABEL!r}: {sorted(blocks)}")
    by_label = {cond["label"]: cond for cond in conditions}
    reference = np.asarray(blocks[REFERENCE_LABEL], dtype=np.float64).mean(axis=0)

    rows: list[dict[str, Any]] = []
    for label, block in blocks.items():
        arr = np.asarray(block, dtype=np.float64)
        cond = by_label.get(label, {"expect": "?", "note": "", "exposure_ms": None})
        sims: list[float] = []
        drifts: list[float] = []
        shape_rows: list[dict[str, Any]] = []
        shape_ok: list[bool] = []
        unknown_shape = 0
        for frame in arr:
            comparison = compare_flat_frames(frame, reference)
            sims.append(float(comparison.similarity))
            drifts.append(float(np.hypot(*comparison.drift_px)))
            shape = getattr(comparison, "shape_comparison", None)
            if shape is None:
                unknown_shape += 1
                continue
            shape_rows.append(shape.to_dict())
            shape_ok.append(bool(shape.within_tolerance))
        peak = [float(np.asarray(frame).max()) for frame in arr]
        # 形状判据还没进生产时 (shape_comparison 缺失) 只按 NCC + 漂移判决, 并把
        # 这一行标成「形状未知」—— 否则全部条件会被判成「变了」, 表看着对实则空。
        shape_gate = bool(shape_ok) and len(shape_ok) == len(sims)
        reuse = bool(
            min(sims) >= min_similarity
            and max(drifts) <= max_drift_px
            and (not shape_gate or all(shape_ok))
        )
        row: dict[str, Any] = {
            "label": label,
            "expect": cond["expect"],
            "note": cond["note"],
            "exposure_ms": cond["exposure_ms"],
            "n": int(arr.shape[0]),
            "peak_max": max(peak),
            "peak_median": float(np.median(peak)),
            "saturated_px": _saturation(arr),
            "similarity": {
                "min": min(sims),
                "median": float(np.median(sims)),
                "max": max(sims),
            },
            "abs_drift_px": {"min": min(drifts), "max": max(drifts)},
            "shape_unknown_frames": unknown_shape,
            "reuse": reuse,
            # below_floor: 幅度低于探针可测下限, 任何判决都接受 → agrees=None,
            # 既不算一致也不算 MISS (见模块 docstring「低于下限的物理依据」)。
            "agrees": (
                None
                if cond["expect"] == "below_floor"
                else bool(reuse == (cond["expect"] == "same"))
            ),
        }
        if shape_rows:
            row["shape"] = {
                "any_within_tolerance": any(shape_ok),
                "all_within_tolerance": all(shape_ok),
                "size_ratio_median": float(
                    np.median([item["size_ratio"] for item in shape_rows])
                ),
                "aspect_mismatch_median": float(
                    np.median([item["aspect_mismatch"] for item in shape_rows])
                ),
                "axis_tilt_deg_median": float(
                    np.median([item["axis_tilt_deg"] for item in shape_rows])
                ),
            }
        rows.append(row)

    mismatches = [
        {"label": row["label"], "expect": row["expect"], "reuse": row["reuse"]}
        for row in rows
        if row["agrees"] is False  # below_floor 的 agrees=None, 不算 MISS
    ]
    confusion = {
        "expected_same": sum(1 for row in rows if row["expect"] == "same"),
        "expected_diff": sum(1 for row in rows if row["expect"] == "diff"),
        "below_floor": sum(1 for row in rows if row["expect"] == "below_floor"),
        "reuse_same": sum(1 for row in rows if row["reuse"]),
        "reuse_diff": sum(1 for row in rows if not row["reuse"]),
        "agrees": sum(1 for row in rows if row["agrees"]),
        "total": len(rows),
    }
    # consistent = 判对 + 低于下限 (后者不计对错, 但刺激构造正确, 合并报「自洽」)。
    confusion["consistent"] = confusion["agrees"] + confusion["below_floor"]
    return {
        "reference_label": REFERENCE_LABEL,
        "reference_frames": int(np.asarray(blocks[REFERENCE_LABEL]).shape[0]),
        "min_similarity": min_similarity,
        "max_drift_px": max_drift_px,
        "rows": rows,
        "mismatches": mismatches,
        "confusion": confusion,
    }


def format_table(analysis: dict[str, Any]) -> str:
    """把分析结果排成一张等宽表 (终端可读)。"""
    header = (
        f"{'condition':<20s}{'期望':<6s}{'n':>3s}{'sim_min':>9s}{'sim_med':>9s}"
        f"{'drift':>8s}{'size':>7s}{'aspect':>8s}{'tilt°':>7s}{'判决':>6s}{'一致':>6s}"
    )
    lines = [header, "-" * len(header)]
    for row in analysis["rows"]:
        shape = row.get("shape") or {}
        # below_floor 在「期望」「一致」两列都缩写成 floor, 保持 6 字宽对齐。
        expect_disp = "floor" if row["expect"] == "below_floor" else row["expect"]
        agree_disp = (
            "floor" if row["agrees"] is None else "OK" if row["agrees"] else "MISS"
        )
        lines.append(
            f"{row['label']:<20s}{expect_disp:<6s}{row['n']:>3d}"
            f"{row['similarity']['min']:>9.4f}{row['similarity']['median']:>9.4f}"
            f"{row['abs_drift_px']['max']:>8.2f}"
            f"{shape.get('size_ratio_median', float('nan')):>7.3f}"
            f"{shape.get('aspect_mismatch_median', float('nan')):>8.3f}"
            f"{shape.get('axis_tilt_deg_median', float('nan')):>7.2f}"
            f"{'same' if row['reuse'] else 'diff':>6s}"
            f"{agree_disp:>6s}"
        )
    confusion = analysis["confusion"]
    lines.append("")
    lines.append(
        f"混淆表: 期望 same {confusion['expected_same']} / diff "
        f"{confusion['expected_diff']} / floor {confusion['below_floor']}; "
        f"判 same {confusion['reuse_same']} / diff {confusion['reuse_diff']}; "
        f"一致 {confusion['agrees']}/{confusion['total']} "
        f"(floor {confusion['below_floor']}, 合计 {confusion['consistent']}"
        f"/{confusion['total']})"
    )
    if confusion["below_floor"]:
        lines.append("  (floor = below_floor: 幅度低于探针可测下限, 不计对错)")
    if analysis["mismatches"]:
        for item in analysis["mismatches"]:
            lines.append(
                f"  MISS {item['label']}: 期望 {item['expect']} "
                f"判 {'same' if item['reuse'] else 'diff'}"
            )
    else:
        lines.append(f"  无 MISS (自洽 {confusion['consistent']}/{confusion['total']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """入口: 默认只打印计划; ``--execute`` 实验; ``--analyse`` 只复算。"""
    from ao_shaping.tools.slm.transfer.optical_probe import (
        DEFAULT_MAX_ZERO_DRIFT_PX,
        DEFAULT_MIN_FLAT_SIMILARITY,
    )

    args = parse_args(argv)

    if args.analyse:
        run_dir = Path(args.analyse)
        with np.load(run_dir / "frames.npz") as payload:
            blocks = {key: np.asarray(payload[key]) for key in payload.files}
        conditions = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
        # plan.json 里的 expect 是 run 当时 (改标前) 的旧标签; 用当前 expected_by_label()
        # 按标签覆盖, 否则复算会拿旧标签重新计分, below_floor / tilt-14.8px 改标不生效。
        fresh_expects = expected_by_label()
        for cond in conditions:
            if cond["label"] in fresh_expects:
                cond["expect"] = fresh_expects[cond["label"]]
        analysis = analyse_blocks(
            blocks,
            conditions,
            min_similarity=DEFAULT_MIN_FLAT_SIMILARITY,
            max_drift_px=DEFAULT_MAX_ZERO_DRIFT_PX,
        )
        print(format_table(analysis))
        (run_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"复算完成: {run_dir / 'analysis.json'}")
        return 0 if not analysis["mismatches"] else 3

    conditions = build_conditions()
    print(describe_plan(conditions, args.exposure_ms))

    if not args.execute:
        print("\n[离线] 只打印计划, 未打开设备; 加 --execute 实验。")
        return 0

    out = Path(args.out_dir) if args.out_dir else default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    precheck = precheck_bench(float(args.wait_until_idle))

    result = run_device(args, out)
    result["precheck"] = precheck

    with np.load(out / "frames.npz") as payload:
        blocks = {key: np.asarray(payload[key]) for key in payload.files}
    analysis = analyse_blocks(
        blocks,
        conditions,
        min_similarity=DEFAULT_MIN_FLAT_SIMILARITY,
        max_drift_px=DEFAULT_MAX_ZERO_DRIFT_PX,
    )
    result["analysis"] = analysis
    (out / "analysis.json").write_text(
        json.dumps(analysis, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    (out / "run.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print()
    print(format_table(analysis))
    logger.info(f"输出: {out}")
    return 0 if not analysis["mismatches"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
