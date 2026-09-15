"""SLM Zernike 模式 → WFS 读数 响应矩阵标定 (Zernike 模式法波前矫正).

在标定后的最优 SLM 相位中心 shift 下, 对每个 SLM Zernike 模式施加 ±A 的推拉扰动,
测量 WFS 的 Zernike 读数增量, 构建响应矩阵:

    matrix[j, i] = Δ(WFS 系数 j) / Δ(SLM 模式 i 幅度)      [λ / λ]

该矩阵求逆即得 Zernike 模式法波前矫正的控制律 (WFS 测得的像差 → SLM 需加载的模式)。

相位生成走 `PatternHelper.generate_zernike_polynomial()` + `display_phase()`
(与 `multi_slm_controller` GUI 同一调用链, 已在 WFS 官方软件验证正确)。

⚠️ 重要索引约定 (官方手册 + 硬件实测确认, 非标准 Noll 1976)
--------------------------------------------------------
`ThorlabWFS.get_zernike()` 返回 67 长数组, DLL 填 `coeff[1..66]` (index 0 未用)。
其排序为**顺序 m 枚举** (`m = -n..+n`):

    [1](0,0) [2](1,-1) [3](1,1) [4](2,-2) [5](2,0)defocus [6](2,2)
    [7](3,-3) [8](3,-1) [9](3,1)coma [10](3,3)
    [11](4,-4) [12](4,-2) [13](4,0)spherical [14](4,2) [15](4,4)

手册佐证: `roCMm` "derived from Zernike coefficient Z[5]" (球面波 RoC ← defocus)。
实测佐证: 加载 (2,0)→[5], (2,2)→[6], (3,1)→[9], (4,0)→[13] 全部吻合。

用法
----
    python -m ao_shaping.tools.slm.slm_zernike_response
    python -m ao_shaping.tools.slm.slm_zernike_response --n-max 4 --amplitude-rad 5
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import click
import numpy as np
from loguru import logger

from ao_shaping.drivers.slm import Santec
from ao_shaping.drivers.wfs import ThorlabWFS
from ao_shaping.optimizer.wf.zernike_response_matrix import (
    ZernikeResponseMatrixResult,
    load_zernike_response_matrix,
    save_zernike_response_matrix,
)
from ao_shaping.tools.slm.slm_zernike_common import (
    DEFAULT_EXPOSURE_MS,
    DLL_ZERNIKE_NAMES,
    DLL_ZERNIKE_ORDER,
    MAX_EXPOSURE_MS,
    PANEL_H,
    PANEL_W,
    SETTLE_REDUNDANCY_S,
    WFS_ZERNIKE_ORDER,
    flat_gray,
    make_phase,
    measure_zernike,
    show_phase,
    wfs_validity,
)
from ao_shaping.utils.pattern_helper import PatternHelper

# 兼容旧引用: DLL 顺序 m 枚举索引表 (定义在 slm_zernike_common)
DLL_ZERNIKE = DLL_ZERNIKE_ORDER
DLL_NAMES = DLL_ZERNIKE_NAMES


def _measure_zernike(wfs: ThorlabWFS, n_avg: int, order: int) -> np.ndarray | None:
    """多帧中位数聚合 WFS Zernike 读数 (委托 slm_zernike_common.measure_zernike)."""
    return measure_zernike(wfs, n_avg=n_avg, order=order)


def verify_response_matrix(path: str, top_n: int = 5) -> int:
    """离线校验已保存的响应矩阵: 形状/元数据/对角线/条件数/逆矩阵/闭环反解演示."""
    r = load_zernike_response_matrix(path)
    dc = dict(r.device_config or {})
    click.echo("=" * 72)
    click.echo(f"[VERIFY] {path}")
    click.echo("=" * 72)
    click.echo(f"matrix={r.matrix.shape} variance={r.variance_matrix.shape} "
               f"pinv={None if r.pinv_matrix is None else r.pinv_matrix.shape}")
    click.echo(f"n_max={r.n_max} magnitude={r.magnitude:.4f}λ "
               f"wavelength={r.wavelength_nm}nm n_avg={r.n_averages} n_cycles={r.n_cycles}")
    click.echo(f"excluded_piston={r.excluded_piston} "
               f"excluded_tip_tilt={r.excluded_tip_tilt}")
    click.echo(f"mean_variance={r.mean_variance:.3e} "
               f"condition_number={np.linalg.cond(r.matrix):.3f}")
    click.echo(f"shift={dc.get('shift_x', dc.get('shift'))} "
               f"mode_ids={dc.get('slm_mode_ids_dll')}")

    ids = dc.get("slm_mode_ids_dll") or list(range(2, r.matrix.shape[1] + 2))
    click.echo(f"\n--- 对角线检查 (SLM 模式 i → WFS 同索引) ---")
    diag_ok = 0
    for col, midx in enumerate(ids):
        row = midx - 1
        if row >= r.matrix.shape[0]:
            continue
        colmax = int(np.argmax(np.abs(r.matrix[:, col])))
        is_diag = (colmax == row)
        diag_ok += int(is_diag)
        click.echo(f"  SLM[{midx:2d}] → WFS[{row + 1:2d}] = {r.matrix[row, col]:+.3f}  "
                   f"(列主导 [{colmax + 1:2d}] = {r.matrix[colmax, col]:+.3f}) "
                   f"{'OK' if is_diag else 'MISMATCH'}")
    click.echo(f"对角线主导: {diag_ok}/{len(ids)}")

    if r.pinv_matrix is None:
        click.echo("\n[WARN] 无逆矩阵 (pinv/lstsq), 无法演示反解")
        return 0 if diag_ok == len(ids) else 1

    w = np.zeros(r.matrix.shape[0])
    w[4] = 0.30    # DLL [5] defocus
    w[8] = -0.20   # DLL [9] coma(3,1)
    c = r.pinv_matrix @ w
    resid = r.matrix @ c - w
    red = 100 * (1 - float(np.linalg.norm(resid)) / float(np.linalg.norm(w)))
    click.echo(f"\n--- 闭环反解演示 (合成像差 [5]=+0.30, [9]=−0.20) ---")
    for i, mid in enumerate(ids):
        if abs(c[i]) > 1e-3:
            click.echo(f"  SLM[{mid:2d}] = {c[i]:+.4f}")
    click.echo(f"残差 ‖Mc−w‖={np.linalg.norm(resid):.4f} / ‖w‖={np.linalg.norm(w):.4f} "
               f"→ 降低 {red:.1f}%")
    return 0 if (diag_ok == len(ids) and red > 50) else 1


@click.command()
@click.option("--slm-number", type=int, default=1, show_default=True)
@click.option("--slm-wavelength", type=int, default=532, show_default=True, help="SLM 波长 nm")
@click.option("--wfs-exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS, show_default=True,
              help=f"WFS 曝光 ms (必须 ≤ {MAX_EXPOSURE_MS})")
@click.option("--wfs-order", type=int, default=10, show_default=True,
              help="WFS Zernike 拟合阶数 (有效 2..10, 10 → 66 项)")
@click.option("--n-max", type=int, default=4, show_default=True,
              help="扫描的 SLM Zernike 最大阶数 (4 → 15 模式)")
@click.option("--zernike-radius", type=float, default=300.0, show_default=True,
              help="Zernike 归一化半径 px (实测光束半径≈200px; 建议 ≥1.5×光束半径, "
                   "R≈光束半径时大振幅会击穿 WFS 拟合)")
@click.option("--amplitude-rad", type=float, default=5.0, show_default=True,
              help="推拉扰动幅度 rad (Zernike 系数)")
@click.option("--n-avg", type=int, default=2, show_default=True, help="每点 WFS 帧平均")
@click.option("--n-cycles", type=int, default=2, show_default=True, help="推拉循环次数 (估方差)")
@click.option("--shift-x", type=int, default=None, help="SLM shift_x (默认读设备配置)")
@click.option("--shift-y", type=int, default=None, help="SLM shift_y (默认读设备配置)")
@click.option("--exclude-tip-tilt", is_flag=True, default=False,
              help="排除 tip/tilt 模式 (DLL [2],[3])")
@click.option("--settle-extra-s", type=float, default=SETTLE_REDUNDANCY_S,
              show_default=True, help="像素翻转估算之外的冗余等待 (s)")
@click.option("-o", "--output", default=None, help="输出 h5 路径 (默认 data/zernike_response_matrix/)")
@click.option("--verify", "verify_path", default=None,
              help="离线校验已保存的 h5 (不接触硬件); 指定后忽略其他选项")
def main(
    slm_number: int,
    slm_wavelength: int,
    wfs_exposure_ms: float,
    wfs_order: int,
    n_max: int,
    zernike_radius: float,
    amplitude_rad: float,
    n_avg: int,
    n_cycles: int,
    shift_x: int | None,
    shift_y: int | None,
    exclude_tip_tilt: bool,
    settle_extra_s: float,
    output: str | None,
    verify_path: str | None,
) -> int:
    """SLM Zernike 模式 → WFS 读数 响应矩阵标定."""
    if verify_path:
        return verify_response_matrix(verify_path)
    if wfs_exposure_ms > MAX_EXPOSURE_MS:
        raise click.BadParameter(f"WFS 曝光 {wfs_exposure_ms}ms > {MAX_EXPOSURE_MS}ms")
    if not 2 <= wfs_order <= 10:
        raise click.BadParameter("--wfs-order 必须在 2..10 (10 → 66 项)")

    click.echo("=" * 72)
    click.echo("[SLM Zernike 响应矩阵标定] Zernike 模式法波前矫正")
    click.echo("=" * 72)

    n_modes_total = (n_max + 1) * (n_max + 2) // 2      # 含 piston
    mode_ids = list(range(1, n_modes_total + 1))        # DLL 1-based
    mode_ids = [i for i in mode_ids if i != 1]          # 排除 piston
    if exclude_tip_tilt:
        mode_ids = [i for i in mode_ids if i not in (2, 3)]

    slm = Santec(slm_number=slm_number, wavelength=slm_wavelength, video_mode=0)
    wfs = ThorlabWFS(exposure_time=wfs_exposure_ms, use_custom_ref=False)
    ph = PatternHelper(resolution=(PANEL_W, PANEL_H))

    # matrix/variance 在测到基线后按 get_zernike 实际返回长度分配 (67 长: index 0 未用)
    matrix: np.ndarray | None = None
    variance: np.ndarray | None = None
    raw: dict = {"modes": [], "baseline": None}

    try:
        slm.open()
        wfs.open()
        exp = float(wfs.exposure_time)
        assert exp <= MAX_EXPOSURE_MS
        wl, max_gray = slm.get_wavelength_info()
        sx = slm.shift_x if shift_x is None else shift_x
        sy = slm.shift_y if shift_y is None else shift_y
        slm.set_shift(sx, sy)
        click.echo(f"[OK] SLM #{slm._serial_number} {wl}nm 2π={max_gray} shift=({sx},{sy})")

        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        cx, cy, dx, dy = wfs.pupil = wfs.optimize_pupil()
        click.echo(f"[OK] WFS {wfs.serial_num} exp={exp:.3f}ms "
                   f"pupil=({cx:.3f},{cy:.3f})mm d=({dx:.3f},{dy:.3f})mm")

        # 纯平用户参考 → 读数只反映加载相位
        flat = np.full((PANEL_H, PANEL_W), 0, dtype=np.uint16)
        slm.display_data(flat, wait_time_s=0.6)
        time.sleep(0.4)
        wfs.take_image(n_sample=1, dynamicNoiseCut=True)
        wfs.create_default_user_ref()
        wfs.set_ref_plane(custom=True)
        base = _measure_zernike(wfs, n_avg=max(n_avg, 3), order=wfs_order)
        if base is None:
            raise RuntimeError("基线测量失败")
        raw["baseline"] = base.tolist()
        click.echo(f"[BASE] 纯平基线 |z[2..6]|="
                   f"{np.linalg.norm(base[2:7]):.4f}λ (噪声底)")

        # get_zernike 返回 67 长数组 (index 0 未用, DLL 填 coeff[1..N]) → 取 [1:] 作 66 通道
        n_wfs_terms = int(base.size - 1)
        matrix = np.zeros((n_wfs_terms, len(mode_ids)), dtype=np.float64)
        variance = np.zeros_like(matrix)

        amp_waves = amplitude_rad / (2 * np.pi)
        click.echo(f"[INFO] 扫描 {len(mode_ids)} 个模式, R={zernike_radius:.0f}px, "
                   f"A={amplitude_rad} rad ({amp_waves:.3f}λ), "
                   f"推拉 {n_cycles} 循环 × {n_avg} 帧, WFS 阶数 {wfs_order} "
                   f"({n_wfs_terms} 项)")

        for col, midx in enumerate(mode_ids):
            nm = DLL_ZERNIKE[midx - 1]
            cols: list[np.ndarray] = []
            for cyc in range(n_cycles):
                zs: dict[int, np.ndarray | None] = {}
                for sign in (+1, -1):
                    phase = make_phase(ph, {nm: sign * amplitude_rad},
                                       zernike_radius, n_max=n_max)
                    show_phase(slm, phase, settle_extra_s)
                    # WFS 有效性门控: 子孔径光斑丢失时读数不可用
                    # (实测 R≈光束半径 + 大振幅 → DLL 拟合崩溃, |resp| 可暴涨到 10³)
                    val = wfs_validity(wfs)
                    if not val["ok"]:
                        logger.warning("模式 {} cycle {} sign {} 光斑有效比 {:.2f} "
                                       "< 阈值 → 剔除", midx, cyc, sign,
                                       val["valid_ratio"])
                        zs[sign] = None
                        continue
                    z = measure_zernike(wfs, n_avg=n_avg, order=wfs_order)
                    if z is None:
                        logger.warning("模式 {} cycle {} sign {} 无有效读数", midx, cyc, sign)
                        zs[sign] = None
                        continue
                    zs[sign] = z
                z_pos, z_neg = zs.get(+1), zs.get(-1)
                if z_pos is not None and z_neg is not None:
                    # 推拉差分 (λ 单位) / 幅度(λ) → 单位幅度响应
                    cols.append((z_pos - z_neg) / 2.0 / amp_waves)
            if not cols:
                click.echo(f"[WARN] 模式 [{midx}] {nm} 无有效响应, 置零")
                continue
            arr = np.array(cols)[:, 1:]                 # (cycles, n_wfs_terms) 去 index 0
            col_vec = np.median(arr, axis=0)
            assert matrix is not None and variance is not None
            matrix[:, col] = col_vec
            variance[:, col] = np.var(arr, axis=0) if len(cols) > 1 else 0.0
            top = np.argsort(np.abs(col_vec))[::-1][:3]
            tops = ", ".join(f"[{int(k) + 1}]{col_vec[k]:+.3f}" for k in top)
            raw["modes"].append({"dll_index": midx, "nm": list(nm),
                                 "response": col_vec.tolist()})
            click.echo(f"[{col + 1:2d}/{len(mode_ids)}] [{midx:2d}] {str(nm):9s} "
                       f"|resp|={np.linalg.norm(col_vec):7.3f}  主导: {tops}")

        slm.display_data(flat, wait_time_s=0.5)

    except Exception as e:
        logger.exception("标定失败")
        click.echo(f"[FAIL] {type(e).__name__}: {e}")
        return 1
    finally:
        for dev in (wfs, slm):
            try:
                dev.close()
            except Exception:
                pass

    # ---------- 保存 ----------
    if matrix is None or variance is None:
        click.echo("[FAIL] 未获得有效响应矩阵")
        return 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    device_config = {
        "slm_serial": slm._serial_number,
        "wfs_serial": wfs.serial_num,
        "wavelength_nm": slm_wavelength,
        "slm_2pi_gray": max_gray,
        "wfs_exposure_ms": exp,
        "shift_x": int(sx),
        "shift_y": int(sy),
        "pupil_center_mm": [cx, cy],
        "pupil_diameter_mm": [dx, dy],
        "zernike_radius_px": zernike_radius,
        "amplitude_rad": amplitude_rad,
        "amplitude_waves": amp_waves,
        "wfs_zernike_order": wfs_order,
        "slm_mode_ids_dll": mode_ids,
        "slm_mode_nm": [list(DLL_ZERNIKE[i - 1]) for i in mode_ids],
        "zernike_ordering": (
            "DLL 顺序 m 枚举 (m=-n..+n), 非标准 Noll 1976: "
            "[1](0,0) [2](1,-1) [3](1,1) [4](2,-2) [5](2,0)defocus [6](2,2) "
            "[9](3,1)coma [13](4,0)spherical"
        ),
        "matrix_layout": (
            f"matrix[wfs_coeff_index, slm_mode_index]; wfs_coeff_index 0..{n_wfs_terms - 1} "
            "对应 DLL [1..N] (非 Noll)"
        ),
        "method": "push-pull ±A, PatternHelper + display_phase (GUI 同链路)",
        "reference": "custom user ref @ flat phase (shift 已应用)",
    }
    result = ZernikeResponseMatrixResult(
        matrix=matrix,
        variance_matrix=variance,
        deviation_response_matrix=None,
        subaperture_mask=None,
        n_max=n_max,
        magnitude=amp_waves,
        wavelength_nm=slm_wavelength,
        n_averages=n_avg,
        n_cycles=n_cycles,
        timestamp=datetime.now().isoformat(),
        excluded_piston=True,
        excluded_tip_tilt=exclude_tip_tilt,
        device_config=device_config,
    )
    # 逆矩阵 (Zernike 模式法矫正控制律: c = pinv(M) @ w, c=(n_slm_modes,), w=(n_wfs_terms,))
    try:
        result.pinv_matrix = np.linalg.pinv(matrix)
        result.lstsq_matrix = result.pinv_matrix
        device_config["inverse_layout"] = (
            f"pinv_matrix shape {result.pinv_matrix.shape} = (slm_modes, wfs_terms); "
            "c = pinv @ w 给出使 ||M c - w|| 最小的 SLM 模式幅度 (λ)"
        )
    except np.linalg.LinAlgError as e:
        logger.warning("逆矩阵计算失败: {}", e)

    if output:
        out = Path(output)
    else:
        out = Path("data/zernike_response_matrix") / (
            f"zm_slm{slm._serial_number}_wfs{wfs.serial_num}_{slm_wavelength}nm_{ts}.h5"
        )
    save_zernike_response_matrix(result, out)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({"device_config": device_config, "raw": raw,
                                   "matrix": matrix.tolist(),
                                   "variance_matrix": variance.tolist()},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"\n[OK] 响应矩阵: {out}  shape={matrix.shape}")
    click.echo(f"[OK] 原始数据: {sidecar}")
    click.echo(f"[INFO] 条件数={np.linalg.cond(matrix):.2f}, "
               f"平均方差={float(np.mean(variance)):.3e}")
    click.echo("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
