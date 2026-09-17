"""SLM Zernike 工具集 — 共享常量与测量原语.

被 `slm_shift_calib` / `slm_zernike_response` / `slm_zernike_correction` /
`slm_wfs_reference` / `slm_zernike_report` 共同复用, 避免重复实现。

⚠️ 核心索引约定 (官方手册 + 硬件实测双重确认, **非标准 Noll 1976**)
------------------------------------------------------------------
`ThorlabWFS.get_zernike()` 返回 67 长数组, DLL 填 `coeff[1..66]` (index 0 未用),
排序为**顺序 m 枚举** (`m = -n..+n`):

    [1](0,0) [2](1,-1) [3](1,1) [4](2,-2) [5](2,0)defocus [6](2,2)
    [7](3,-3) [8](3,-1) [9](3,1)coma [10](3,3)
    [11](4,-4) [12](4,-2) [13](4,0)spherical [14](4,2) [15](4,4)

- 手册佐证: `roCMm` "derived from Zernike coefficient **Z[5]**" (球面波 RoC ← defocus);
  `arrayZernikeUm` "indices [1..66] are used instead of [0..65]"。
- 实测佐证: 加载 (2,0)→[5], (2,2)→[6], (3,1)→[9], (4,0)→[13]; 响应矩阵 14/14 强对角。

标准 Noll 中 defocus=4、spherical=11 ⇒ 驱动 docstring 的 "Noll 1976 约定" 不准确。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from ao_shaping.drivers.slm.santec import Santec, WavefrontCorrection
from ao_shaping.drivers.wfs import ThorlabWFS
from ao_shaping.tools.slm.slm_scan_analysis import outlier_mask
from ao_shaping.utils.pattern_helper import PatternHelper

PANEL_H, PANEL_W = 1200, 1920
MAX_EXPOSURE_MS = 7.0
DEFAULT_EXPOSURE_MS = 4.0
SETTLE_REDUNDANCY_S = 0.1          # 像素翻转估算之外额外等待
WFS_ZERNIKE_ORDER = 10             # orders 有效值 0=auto 或 2..10 (10 → 66 项); 15 非法!
LAMBDA_UM = 0.532                  # 工作波长 (µm): WFS 系数单位 µm → λ 需 ÷ 此值

# ⚠️ 单位一致性 (2026-09-16 实测定位的矫正失效根因):
#   `get_zernike()` 返回**µm**; 响应矩阵必须与矫正时的 `w` 用**同一单位**。
#   曾因矩阵用原始 µm 构建、而矫正用 `w = z/0.532` (λ), 反解系数被放大 1/0.532 = 1.88×,
#   矫正过驱动 88% → 实测闭环仅 19.7% (离线最优可达 75%)。**一律经 `um_to_waves` 换算。**
UM_TO_WAVES = 1.0 / LAMBDA_UM


def um_to_waves(z: np.ndarray) -> np.ndarray:
    """WFS Zernike 系数 µm → λ (工作波长 532nm)。矩阵与矫正**必须**用同一单位。"""
    return np.asarray(z, dtype=float) * UM_TO_WAVES

# DLL 顺序 m 枚举: 1-based index -> (n, m)。完整 66 项 (n=0..10, 每阶 n+1 项)。
# ⚠️ 曾只到 n=5 (21 项) — 与 mode_ids 2..66 (n_max=10) 不匹配, 二阶以上 IndexError。
DLL_ZERNIKE_ORDER: list[tuple[int, int]] = [
    (0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2),
    (3, -3), (3, -1), (3, 1), (3, 3),
    (4, -4), (4, -2), (4, 0), (4, 2), (4, 4),
    (5, -5), (5, -3), (5, -1), (5, 1), (5, 3), (5, 5),
    (6, -6), (6, -4), (6, -2), (6, 0), (6, 2), (6, 4), (6, 6),
    (7, -7), (7, -5), (7, -3), (7, -1), (7, 1), (7, 3), (7, 5), (7, 7),
    (8, -8), (8, -6), (8, -4), (8, -2), (8, 0), (8, 2), (8, 4), (8, 6), (8, 8),
    (9, -9), (9, -7), (9, -5), (9, -3), (9, -1),
    (9, 1), (9, 3), (9, 5), (9, 7), (9, 9),
    (10, -10), (10, -8), (10, -6), (10, -4), (10, -2),
    (10, 0), (10, 2), (10, 4), (10, 6), (10, 8), (10, 10),
]

DLL_ZERNIKE_NAMES: dict[int, str] = {
    1: "(0,0)piston", 2: "(1,-1)tiltA", 3: "(1,1)tiltB", 4: "(2,-2)astig",
    5: "(2,0)defocus", 6: "(2,2)astig", 7: "(3,-3)trefoil", 8: "(3,-1)coma",
    9: "(3,1)coma", 10: "(3,3)trefoil", 11: "(4,-4)quadrafoil",
    12: "(4,-2)2nd-astig", 13: "(4,0)spherical", 14: "(4,2)2nd-astig",
    15: "(4,4)quadrafoil",
}


def nm_of(index: int) -> tuple[int, int]:
    """DLL 1-based 索引 → (n, m)."""
    return DLL_ZERNIKE_ORDER[index - 1]


def index_of(nm: tuple[int, int]) -> int | None:
    """(n, m) → DLL 1-based 索引 (未收录返回 None)."""
    try:
        return DLL_ZERNIKE_ORDER.index(nm) + 1
    except ValueError:
        return None


def name_of(index: int) -> str:
    return DLL_ZERNIKE_NAMES.get(index, f"idx{index}")


def n_modes_upto(n_max: int) -> int:
    """含 piston 的模式总数 (n_max=4 → 15)."""
    return (n_max + 1) * (n_max + 2) // 2


# ─────────────────────────── 设备参数采集 ───────────────────────────

def collect_device_info(slm: Santec, wfs: ThorlabWFS | None = None,
                        slm_number: int | None = None) -> dict[str, Any]:
    """采集 SLM/WFS 完整设备参数供报告记录 (单点读取失败记 None, 不中断).

    - **SLM**: 序列号 / DisplayName / 固件版本 (DLL·Drive·Option·FPGA) / 工作波长 /
      **最大相位 (2π) 与对应灰度** / 面板分辨率 / shift / 视频模式 / **工作温度 (驱动板+选件板)**
      / 波前矫正启用状态与 CSV / LUT / 当前灰度 / 响应时间与像素翻转上限
    - **WFS**: 序列号 / 设备名·厂商·型号 / **曝光时间** / **pupil (中心+直径)** / MLA 名称
      / 子孔径数 (27×27) / mla_index / 参考平面 (custom?) / 高速模式 / 主增益
    """
    info: dict[str, Any] = {"slm": {}, "wfs": {}}

    def _get(obj: Any, attr: str) -> Any:
        try:
            return getattr(obj, attr)
        except Exception as e:      # noqa: BLE001 - 单点读取失败不应中断
            logger.debug("读取 {}.{} 失败: {}", type(obj).__name__, attr, e)
            return None

    s = info["slm"]
    s["slm_number"] = slm_number if slm_number is not None else _get(slm, "slm_number")
    s["serial_number"] = _get(slm, "serial_number") or _get(slm, "_serial_number")
    s["display_name"] = _get(slm, "display_name")
    s["version"] = _get(slm, "version")            # DLL/Drive/Option/FPGA
    try:
        wl, mg = slm.get_wavelength_info()
        s["wavelength_nm"] = int(wl)
        s["two_pi_gray"] = int(mg)
        s["max_phase_rad"] = 2.0 * float(np.pi)    # 相位调制上限 = 2π
        s["max_phase_waves"] = 1.0
    except Exception as e:                          # noqa: BLE001
        logger.debug("get_wavelength_info 失败: {}", e)
    s["max_grayscale_value"] = _get(slm, "MAX_GRAYSCALE_VALUE")
    pr = _get(slm, "Panel_Res")
    s["panel_res"] = list(pr) if pr else None
    s["shift_x"] = _get(slm, "shift_x")
    s["shift_y"] = _get(slm, "shift_y")
    vm = _get(slm, "video_mode")
    s["video_mode"] = int(vm) if isinstance(vm, (int, np.integer)) else str(vm)
    temp = _get(slm, "temperature")                 # (驱动板, 选件板) °C
    s["temperature_c"] = (list(temp) if isinstance(temp, (tuple, list)) else temp)
    s["correction_enabled"] = _get(slm, "correction_enabled")
    cp = _get(slm, "correction_csv_path")
    s["correction_csv_path"] = str(cp) if cp else None
    ld = _get(slm, "lut_dir")
    s["lut_dir"] = str(ld) if ld else None
    s["lut_loaded"] = _get(slm, "lut") is not None
    s["current_grayscale"] = _get(slm, "current_grayscale")
    s["is_open"] = _get(slm, "is_open")
    s["response_time_ms"] = _get(slm, "Response_time_ms")
    s["max_pixel_flip_ms"] = _get(slm, "MAX_PIXEL_FLIP_TIME_MS")

    if wfs is not None:
        w = info["wfs"]
        try:
            w.update(wfs.get_hardware_info())
        except Exception as e:                      # noqa: BLE001
            logger.debug("wfs.get_hardware_info 失败: {}", e)
        w["exposure_time_ms"] = _get(wfs, "exposure_time")
        try:
            cx, cy, dx, dy = wfs.pupil
            w["pupil_center_mm"] = [float(cx), float(cy)]
            w["pupil_diameter_mm"] = [float(dx), float(dy)]
        except Exception as e:                      # noqa: BLE001
            logger.debug("wfs.pupil 读取失败: {}", e)
        try:
            w["mla_name"] = wfs.get_mla_name()
        except Exception as e:                      # noqa: BLE001
            logger.debug("wfs.get_mla_name 失败: {}", e)
        w["num_spots_x"] = _get(wfs, "num_spots_x")
        w["num_spots_y"] = _get(wfs, "num_spots_y")
        mi = _get(wfs, "mla_index")
        w["mla_index"] = str(mi)
        w["use_custom_ref"] = _get(wfs, "use_custom_ref")
        w["high_speed"] = _get(wfs, "high_speed")
        w["master_gain"] = _get(wfs, "master_gain")
    return info


def safe_pinv(matrix: np.ndarray) -> np.ndarray:
    """对可能含**零列**的响应矩阵求伪逆 (零列对应行置 0, 仅对有效列求逆).

    直接 ``np.linalg.pinv(M)`` 在含零列时条件数会爆到 1e18 (数值秩亏), 不稳定;
    本函数先挑出有效列求伪逆再展开 —— 语义正确 (被线性度门控剔除的模式系数恒为 0)
    且数值稳定。
    """
    m = np.asarray(matrix, dtype=float)
    valid = [i for i in range(m.shape[1]) if np.linalg.norm(m[:, i]) > 0]
    pinv = np.zeros((m.shape[1], m.shape[0]))
    if valid:
        pinv[valid, :] = np.linalg.pinv(m[:, valid])
    return pinv


def effective_cond(matrix: np.ndarray) -> float:
    """有效列上的条件数 (忽略零列; 全零返回 nan)."""
    m = np.asarray(matrix, dtype=float)
    valid = [i for i in range(m.shape[1]) if np.linalg.norm(m[:, i]) > 0]
    if len(valid) < 2:
        return float("nan")
    return float(np.linalg.cond(m[:, valid]))


# ─────────────────────────── 产物命名 ───────────────────────────

def correction_artifact_name(serial: str | None, wavelength_nm: int | None,
                             shift_x: int | None, shift_y: int | None,
                             zernike_radius: float | None,
                             ts: str | None = None,
                             prefix: str = "slm_corr") -> str:
    """构造**可复原**的矫正相位 CSV 文件名.

    文件名内嵌复原所需的关键参数 —— 序列号 · 波长 · shift_x/shift_y ·
    Zernike 半径 · 时间戳, 使产物脱离上下文也能对应回设备与标定条件:

        slm_corr_23020026_532nm_shift105_40_R300_20260916_011500.csv

    Args:
        serial: SLM 序列号 (None → ``unknown``)
        wavelength_nm: 工作波长 nm
        shift_x / shift_y: SLM 平移像素
        zernike_radius: 标定/生成相位所用 Zernike 半径 px (必须与标定时一致)
        ts: 时间戳字符串, 默认当前时间 ``%Y%m%d_%H%M%S``
        prefix: 文件名前缀

    Returns:
        文件名 (不含目录), 扩展名 ``.csv``
    """
    from datetime import datetime as _dt

    stamp = ts or _dt.now().strftime("%Y%m%d_%H%M%S")
    s = serial or "unknown"
    wl = f"{int(wavelength_nm)}nm" if wavelength_nm else "unknownnm"
    sx = 0 if shift_x is None else int(shift_x)
    sy = 0 if shift_y is None else int(shift_y)
    r = f"R{int(round(float(zernike_radius)))}" if zernike_radius else "Rna"
    return f"{prefix}_{s}_{wl}_shift{sx}_{sy}_{r}_{stamp}.csv"


def _jsonable(obj: Any) -> Any:
    """递归将 dict 键归一为 JSON 允许的类型.

    Zernike 系数字典的键是 ``(n, m)`` tuple, `json.dumps(default=str)` 只兜底
    **value**, 对 tuple **key** 仍抛 TypeError ("keys must be str, int, float, bool
    or None")。sidecar 序列化前统一 str 化非 JSON 键, 防止离线矫正等场景写入失败。
    """
    if isinstance(obj, dict):
        return {
            (k if isinstance(k, (str, int, float, bool)) and k is not None else str(k)):
            _jsonable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def export_correction_csv(slm: Santec, phase_rad: np.ndarray,
                          meta: dict[str, Any],
                          out_dir: str | Path = "data/slm_corrections",
                          ts: str | None = None,
                          prefix: str = "slm_corr") -> tuple[Path, Path]:
    """用**驱动自带**的 `Santec.save_phase_to_csv` 导出矫正相位, 并写同名 sidecar JSON.

    文件名内嵌可复原信息 (见 :func:`correction_artifact_name`); sidecar ``.json``
    记录完整元数据 (设备参数 / 矩阵 / 系数 / 半径 / shift / 残差), 便于复现与审计。

    Args:
        slm: 已打开的 Santec 实例 (仅用于取序列号/波长/shift)
        phase_rad: **弧度制**矫正相位, shape 必须为 (1200, 1920)
        meta: 随 sidecar 落盘的元数据 (设备/矩阵/系数/指标等)
        out_dir: 输出目录
        ts / prefix: 传给 :func:`correction_artifact_name`

    Returns:
        (csv_path, json_path)
    """
    import json as _json
    from datetime import datetime as _dt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = ts or _dt.now().strftime("%Y%m%d_%H%M%S")
    name = correction_artifact_name(
        serial=getattr(slm, "_serial_number", None) or getattr(slm, "serial_number", None),
        wavelength_nm=getattr(slm, "wavelength", None),
        shift_x=getattr(slm, "shift_x", None),
        shift_y=getattr(slm, "shift_y", None),
        zernike_radius=meta.get("zernike_radius_px"),
        ts=stamp, prefix=prefix,
    )
    csv_path = out / name
    # 驱动自带导出 (弧度制 CSV, 保留 Y/X 行列索引)
    Santec.save_phase_to_csv(phase_rad, csv_path)

    sidecar = {
        "csv_file": name,
        "exported_at": _dt.now().isoformat(),
        "phase_units": "radians (Santec.save_phase_to_csv 输出; 重新加载时按弧度读取, "
                       "勿走 load_gray_from_csv/csv_to_phase —— 那两者只接受 0..1023 灰度)",
        "phase_shape": list(np.asarray(phase_rad).shape),
        "phase_range_rad": [float(np.min(phase_rad)), float(np.max(phase_rad))],
        **meta,
    }
    json_path = csv_path.with_suffix(".json")
    json_path.write_text(_json.dumps(_jsonable(sidecar), indent=2, ensure_ascii=False,
                                     default=str), encoding="utf-8")
    return csv_path, json_path


def save_driver_correction_csv(gray_offsets: np.ndarray,
                               dest: str | Path) -> None:
    """按驱动 `Santec.load_gray_from_csv` 的格式导出灰度(偏移)矫正 CSV.

    薄 wrapper —— **唯一实现在 `WavefrontCorrection.save_gray_correction_csv`**
    (CSV I/O 单一真源, 2026-09-16 导出契约):

    - **不含 shift**: 只写面板坐标逐像素偏移, 平移由官方软件 / 驱动
      ``Santec.apply_shift`` 在显示时单独应用;
    - **满量程 2π = 1023**: 取设备常量 ``slm200_constants.get_max_grayscale()``,
      严禁用波长相关 ``two_pi_gray`` (如 532nm→998), 否则官方软件按 1023
      换算时矫正幅度被缩放 ``1023/two_pi_gray ≈ 1.025×``。

    导出格式与 Santec 出厂矫正文件 (``Wavefront_correction_Data_*.csv``) 一致:

    - 首行: ``Y/X,0,1,...,1919`` (列索引, 首字段固定 ``Y/X``)
    - 数据行: ``行索引, g_{i,0},...,g_{i,1919}`` (共 1200 行 × 1921 字段)
    - 数据区为 **0..1023 整数灰度偏移** (uint16), shape ``(1200, 1920)``

    语义 (与 `WavefrontCorrection.map_error` 的加法一致): 导出值 = 待**叠加**
    到任意显示灰度上的矫正量, ``displayed = mod(grayscale + corr, max_gray + 1)``。
    驱动侧加载走 **identity 读取**: ``WavefrontCorrection(csv_path, calc_fn=lambda
    raw: raw.astype(np.float64))``, 切勿套默认余弦拟合 ``_default_calc``。

    Args:
        gray_offsets: (1200, 1920) 灰度偏移, 数值须在 0..1023 (10 位)
        dest: 输出 CSV 路径 (父目录自动创建)

    Raises:
        ValueError: shape 或取值范围不合法
    """
    WavefrontCorrection.save_gray_correction_csv(gray_offsets, dest)


# ─────────────────────────── 相位生成 / 下发 ───────────────────────────

def make_phase(ph: PatternHelper, coefficients: dict[tuple[int, int], float],
               radius: float, n_max: int = 4) -> np.ndarray:
    """Zernike 系数 (rad) → 弧度相位图. 走 PatternHelper (与 GUI 同链路)."""
    return ph.generate_zernike_polynomial(
        coefficients=coefficients, radius=float(radius), n_max=max(n_max, 4))


def flat_gray() -> np.ndarray:
    """纯平相位灰度 (gray=0). 扁平相位必须直接发 uint16, 严禁走 create_phase_from_array."""
    return np.full((PANEL_H, PANEL_W), 0, dtype=np.uint16)


def show_phase(slm: Santec, phase_rad: np.ndarray,
               extra_sleep: float = SETTLE_REDUNDANCY_S) -> None:
    """下发相位: ``wait_time_s=None`` 自动按像素翻转估算等待 + ``extra_sleep`` 冗余.

    估算见 ``Santec._estimate_pixel_flip_wait``:
    ``wait = max_gray_change / _max_gray × MAX_PIXEL_FLIP_TIME_S`` (上限 200 ms)。
    """
    slm.display_phase(phase_rad, wait_time_s=None)
    if extra_sleep > 0:
        time.sleep(extra_sleep)


# ─────────────────────────── 测量 ───────────────────────────

def measure_zernike(wfs: ThorlabWFS, n_avg: int = 3,
                    order: int = WFS_ZERNIKE_ORDER) -> np.ndarray | None:
    """多帧中位数聚合 WFS Zernike 读数 (µm, 67 长, index 0 未用).

    中位数聚合抑制单帧光子噪声/斑点抖动 (小幅度端尤其敏感)。
    """
    rows: list[np.ndarray] = []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        try:
            z = wfs.get_zernike(zernike_order=order)
        except Exception as e:
            logger.debug("get_zernike 失败: {}", e)
            continue
        if z is None or not np.isfinite(z).all():
            continue
        rows.append(np.asarray(z, dtype=float))
    return None if not rows else np.median(np.array(rows), axis=0)


def measure_wavefront(wfs: ThorlabWFS, n_avg: int = 3) -> tuple[np.ndarray, dict] | None:
    """多帧中位数波前 + 平均 stats (keys: rms/diff/mean/max/min)."""
    wfs_, stats = [], []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        wf, st = wfs.get_wavefront(cancel_tile=False)
        wfs_.append(wf)
        stats.append(st)
    if not wfs_:
        return None
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.array(wfs_), axis=0)
    mean_stats = {k: float(np.mean([s.get(k, np.nan) for s in stats]))
                  for k in ("rms", "diff", "mean", "max", "min")}
    return med, mean_stats


def added_tilt(z: np.ndarray, base: np.ndarray) -> tuple[float, np.ndarray]:
    """相对基线的附加倾斜 (λ): DLL [2],[3] 两分量, 返回 (范数, 分量)."""
    t = z[2:4] / 0.532
    b = base[2:4] / 0.532
    d = t - b
    return float(np.linalg.norm(d)), d


def measure_tilt_defocus(
    wfs: ThorlabWFS, n_avg: int = 3, zernike_order: int = 10
) -> tuple[np.ndarray | None, float | None]:
    """多帧中位数聚合, 返回 ``(z_tilt[2] (λ, Noll 2/3), z_defocus (λ, Noll 4))``.

    Verbatim from ``slm_shift_calib.measure_tilt`` (shift 标定共用原语)。
    单帧失败不致命, 交由中位数容忍。

    Args:
        wfs: 已打开的 ThorlabWFS 实例
        n_avg: 平均帧数
        zernike_order: WFS Zernike 阶数 (默认 10 → 66 项)

    Returns:
        ``(zernike, defocus)``: 中位数聚合的 tip/tilt 数组 (λ, 2 分量) 与
        defocus (λ); 全部帧失败时 ``(None, None)``。
    """
    tips: list[np.ndarray] = []
    defoci: list[float] = []
    for _ in range(n_avg):
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        try:
            z_um = wfs.get_zernike(zernike_order=zernike_order)
        except Exception as e:  # 单帧失败不致命, 交由中位数容忍
            logger.debug("get_zernike 失败: {}", e)
            continue
        if z_um is None or not np.isfinite(z_um[2:5]).all():
            continue
        tips.append(np.asarray(z_um[2:4], dtype=float) / 0.532)  # Noll 2,3 = tip/tilt
        defoci.append(float(z_um[4]) / 0.532)  # Noll 4 = defocus
    if not tips:
        return None, None
    return np.median(np.array(tips), axis=0), float(np.median(defoci))


# ─────────────────────────── 自动定标 ───────────────────────────

def diagnose_beam_radius(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
                         radii: list[float], amp_rad: float = 20.0,
                         n_avg: int = 3, extra_sleep: float = SETTLE_REDUNDANCY_S,
                         outlier_factor: float = 10.0,
                         verbose: bool = True) -> tuple[float, list[dict]]:
    """Zernike R 扫描 → WFS defocus 响应最大者 ≈ **光束在 SLM 上的半径**.

    原理: 半径 R 的 defocus 归一化后, 光束 (半径 r_beam) 感受到的相位幅度
    ∝ A·(r_beam/R)² → R 越大响应越小; R < r_beam 时图案被裁切。

    ⚠️ **异常剔除 (实测必需)**: R < 光束半径 时 defocus 盘裁切光束 → 子孔径光斑丢失
    → DLL Zernike 拟合崩溃, `defocus` 读数可暴涨到 10³ 量级 (实测 R=120 → +1266.73λ),
    若直接取 `max(abs)` 会**误选到该异常点** (实测导致扫描半径 180/240 而非 300/400)。
    故先用中位数 × ``outlier_factor`` 剔除异常, 再在剩余点中取响应最大者。
    """
    out: list[dict] = []
    for r in radii:
        phase = make_phase(ph, {(2, 0): amp_rad}, float(r), n_max=5)
        show_phase(slm, phase, extra_sleep)
        z = measure_zernike(wfs, n_avg)
        if z is None:
            if verbose:
                print(f"   R={r:5.0f}px  无有效读数")
            continue
        d = float(z[5]) / 0.532           # DLL [5] = (2,0) defocus
        out.append({"radius": float(r), "defocus_lam": d})
        if verbose:
            print(f"   R={r:5.0f}px  defocus[5]={d:+.4f}λ")
    if not out:
        raise RuntimeError("光束半径诊断无有效数据")

    # 异常剔除 (R < 光束半径 → 盘裁切 → 拟合崩溃)
    mags = np.array([abs(r["defocus_lam"]) for r in out], dtype=float)
    med = float(np.median(mags))
    keep = outlier_mask(mags, outlier_factor)
    if not keep.all():
        for r, kp in zip(out, keep):
            if not kp:
                print(f"   [剔除异常] R={r['radius']:.0f}px "
                      f"defocus={r['defocus_lam']:+.1f}λ (> {outlier_factor}×中位数 "
                      f"{med:.3f}λ) — 盘裁切光束致拟合崩溃")
        out = [r for r, kp in zip(out, keep) if kp]
    if not out:
        raise RuntimeError("光束半径诊断全部为异常值")

    best = max(out, key=lambda r: abs(r["defocus_lam"]))
    if verbose:
        print(f"   → 光束半径 ≈ {best['radius']:.0f}px (剔除后 {len(out)} 个有效点)")
    return float(best["radius"]), out


def calibrate_center_shift(slm: Santec, wfs: ThorlabWFS, ph: PatternHelper,
                           r_beam: float, amp_rad: float = 20.0,
                           coarse_half: int = 300, coarse_step: int = 100,
                           fine_span: int = 40, fine_step: int = 5,
                           n_avg: int = 3,
                           extra_sleep: float = SETTLE_REDUNDANCY_S,
                           limit: int = 500, verbose: bool = True
                           ) -> tuple[int, int, list[dict]]:
    """defocus 零点法测光束中心偏移 → shift_x/shift_y (轴无关判据 ‖Δz_tilt‖).

    图案平移 ``(sx,sy)`` 后光束感受 ``P(b−s+ξ)``; 对 defocus 有梯度 ``∝2D(b−s)``
    → tip/tilt 关于 shift 线性, 零点即 ``s=b``。判据用**相对纯平的附加倾斜**。

    两级扫描: 粗扫 ``±coarse_half`` 步长 ``coarse_step`` → 细扫
    ``±fine_span`` 步长 ``fine_step`` (默认 5px, 提高定位精度)。
    """
    r_use = max(r_beam * 3.0, 450.0)      # 盘需足够大以免平移裁切光束
    phase = make_phase(ph, {(2, 0): amp_rad}, r_use, n_max=5)
    slm.set_shift(0, 0)
    slm.display_data(flat_gray(), wait_time_s=0.5)
    time.sleep(0.3)
    base = measure_zernike(wfs, max(n_avg, 3))
    if base is None:
        raise RuntimeError("基线测量失败")

    scan: list[dict] = []

    def eval_shift(sx: int, sy: int) -> float:
        slm.set_shift(int(np.clip(sx, -limit, limit)), int(np.clip(sy, -limit, limit)))
        show_phase(slm, phase, extra_sleep)
        z = measure_zernike(wfs, n_avg)
        if z is None:
            return float("inf")
        m, _ = added_tilt(z, base)
        return m

    def scan_axis(axis: str, values, other: int) -> float:
        best_v, best_m = float(values[0]), float("inf")
        for v in values:
            sx_ = int(v) if axis == "x" else int(other)
            sy_ = int(v) if axis == "y" else int(other)
            m = eval_shift(sx_, sy_)
            scan.append({"axis": axis, "shift": float(v), "added_norm": float(m)})
            if m < best_m:
                best_m, best_v = m, float(v)
        if verbose:
            print(f"   {axis} 轴最优 {best_v:.0f} (‖Δ‖={best_m:.4f}λ)")
        return best_v

    def fine_around(center: float) -> list[float]:
        return [center + d for d in range(-fine_span, fine_span + 1, fine_step)]

    coarse = list(range(-coarse_half, coarse_half + 1, coarse_step))
    sx0 = scan_axis("x", coarse, 0)
    sx = scan_axis("x", fine_around(sx0), 0)
    sy0 = scan_axis("y", coarse, int(sx))
    sy = scan_axis("y", fine_around(sy0), int(sx))
    if verbose:
        print(f"   → shift=({int(sx)},{int(sy)})")
    return int(sx), int(sy), scan


# ─────────────────────────── 有效性门控 ───────────────────────────

def wfs_validity(wfs: ThorlabWFS, n_avg: int = 1,
                 min_spot_ratio: float = 0.5) -> dict[str, Any]:
    """WFS 测量有效性门控 — 防止大幅度模式下子孔径光斑丢失导致的垃圾拟合.

    返回 dict: valid_ratio / n_valid / n_total / ok。
    ``ok=False`` 时该点读数应剔除或重测 (实测 R≈光束半径 + 大振幅时 DLL 拟合崩溃,
    ``|resp|`` 可暴涨到 10³ 量级)。
    """
    wfs.take_image(n_sample=1, dynamicNoiseCut=True)
    intensity, _ = wfs.get_spots_statics()
    arr = np.asarray(intensity, dtype=float)
    total = int(arr.size)
    valid = int(np.isfinite(arr).sum())
    ratio = valid / total if total else 0.0
    return {"valid_ratio": ratio, "n_valid": valid, "n_total": total,
            "ok": bool(ratio >= min_spot_ratio)}



# ─────────────────────────── 线性度指标 ───────────────────────────

def linearity_metrics(responses: list[np.ndarray], amps: list[float],
                      cv_threshold: float = 0.15, cos_threshold: float = 0.9
                      ) -> dict[str, Any]:
    """推拉响应向量的线性度 (**正确判据**).

    响应向量 ``resp = (z₊ − z₋)/2/a_waves`` **已按单位幅度归一化** → 线性响应表现为
    ``|resp|`` **恒定**。故不能用 slope/R² (线性时 slope≈0, R² 无意义), 而应用:

    - ``cv`` = std(|resp|)/mean(|resp|)  —— 幅度一致性 (越小越线性)
    - ``cos_min`` = 各幅度对之间响应向量夹角余弦最小值 —— 方向一致性
    - ``ok`` = cv < cv_threshold 且 cos_min > cos_threshold

    Args:
        responses: 各幅度下的响应向量列表 (等长)
        amps: 对应幅度 (rad), 仅用于排序

    Returns:
        dict: norms/cv/cos_min/ok/n_amps
    """
    order = np.argsort(np.asarray(amps, dtype=float))
    vecs = [np.asarray(responses[i], dtype=float) for i in order]
    norms = np.array([float(np.linalg.norm(v)) for v in vecs])
    mean_n = float(np.mean(norms)) if len(norms) else 0.0
    cv = float(np.std(norms) / mean_n) if mean_n > 0 else float("inf")
    cos_min = 1.0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            ni, nj = float(np.linalg.norm(vecs[i])), float(np.linalg.norm(vecs[j]))
            if ni > 0 and nj > 0:
                cos_min = min(cos_min, float(vecs[i] @ vecs[j] / (ni * nj)))
    ok = bool(cv < cv_threshold and cos_min > cos_threshold and len(vecs) >= 2)
    return {"norms": norms.tolist(), "cv": cv, "cos_min": cos_min,
            "n_amps": len(vecs), "ok": ok}
