"""从灰度响应 CSV 拟合 SLM 灰度→振幅耦合, 产出 2f 数字孪生可消费的配置。

输入是实验系统上 ``python -m ao_shaping.tools.slm.calibration`` 的灰度响应命令
(``ao_shaping/tools/slm/calibration/gray_response.py``) 写出的 CSV。本工具**完全
离线**, 不碰任何硬件。

输出是 ``bench_twin_calibrate.save_fit`` 约定的 JSON 形状 (顶层 ``{"config": {...}}``),
可被 ``load_fit`` 直接读回, 因此可以原样传给
``scripts/slm/gsnet_transfer.py --twin-config`` 与
``scripts/slm/export_gsnet_phase.py --twin-config``。

物理口径 (强度 vs 振幅 —— 硬件操作员必须能核对的这一条):
    孪生 (``drivers/sim/bench_twin.py``) 的振幅耦合链是
        灰度 g → I(g) = 1 + c·sin(2π·g/levels + ψ)   (灰度域 0 级**归一化强度**)
                 η(g) = sqrt(max(I(g), 0))            (瞳面上的**场振幅**因子)
    对**平相位**面板, η 与 φ 在面板上处处相同, 远场每一格都按 |η|² = I(g) 整体
    缩放 —— 所以相机测到的亮度 (总能量 ``sum_brightness``, 或峰值
    ``max_brightness``) **正比于 I(g) 本身, 不是它的平方**。这正是孪生定义
    ``I`` 的方式: η 是振幅因子, FFT 取模平方之后回到强度, 平方已经在孪生内部
    抵消。因此拟合模型与 ``gray_response.py`` 完全同形::

        y(g) = offset + amplitude · sin(2π·g/period + phase)

    除以 offset 即得孪生两参数::

        c  = amplitude / offset        ψ = phase

    该换算仅在 ``period == levels`` (灰度域一个周期 = 器件的 2π 灰度数) 时严格
    成立; 拟合周期偏离 ``--levels`` 超过几个百分点时本工具会**大声警告**。

已知系统性偏差 (诚实口径): ``offset`` 吸收了相机的暗电流/读噪声本底 pedestal,
而 pedestal 不随灰度调制, 所以 ``c = amplitude/offset`` 是**低估** (offset 偏大)。
``sum_brightness`` 是总收集能量, 是振幅响应的物理正确量 (默认);
``max_brightness`` 是峰值, 对光斑形状漂移更敏感, 只在光斑稳定时可作旁证。

判据对照: 仓库调制检测的合格判据是 ``rel_spread = (max - min)/mean > 0.15``,
对单一次谐波这蕴含 ``2c > 0.15 即 c ≳ 0.075``; 本工具打印实测/拟合曲线的
rel_spread 与对应 c, 供操作员核对。

退出码: 0 = 正常; 1 = CSV 不可用或拟合失败 (**不会**静默退回 c = 0);
2 = ``--check`` 预测与实测不一致超出容差。

用法示例 (实验系统采完 CSV 后离线分析)::

    python scripts/slm/fit_slm_amplitude_coupling.py --csv data/slm_gray_response.csv
    python scripts/slm/fit_slm_amplitude_coupling.py --csv data/slm_gray_response.csv \\
        --metric max --fix-period --base-config data/debug/bench_twin_fit/<ts>/fit.json
    python scripts/slm/fit_slm_amplitude_coupling.py --csv data/slm_gray_response.csv \\
        --check data/debug/slm_amplitude_coupling/<ts>/twin_config.json --plot /tmp/fit.png
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIBS = ROOT / "libs"
for _p in (SRC, LIBS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ao_shaping.drivers.sim.bench_twin import (  # noqa: E402
    BenchTwin,
    BenchTwinConfig,
)

# 器件在 1064 nm 的 2π 灰度数由 EEPROM 查询得到 (993); 1023 是 10-bit 满量程,
# **不是**相位周期数 —— 孪生的灰度域周期必须用前者。
DEFAULT_LEVELS = 993
# 拟合周期与 --levels 允许的最大灰度域偏差 (相对值); 超过即警告。
PERIOD_LEVELS_TOL = 0.02
# 仓库调制检测判据: rel_spread > 0.15 ⇒ c ≳ 0.075 (0 级展平 = 2c)。
VERDICT_REL_SPREAD = 0.15

METRIC_COLUMNS = {
    "sum": "sum_brightness",
    "max": "max_brightness",
    "bucket": "bucket_sum",
}


@dataclass
class GrayResponse:
    """一份灰度响应 CSV 的解析结果。

    Attributes:
        gray: 灰度码列 (float, 已按灰度升序)。
        y: 所选亮度列 (与 ``metric`` 对应)。
        slm_max_gray: CSV 里记录的器件最大灰度 (10-bit 满量程, 非 2π 数)。
        metric: 实际拟合的亮度列名。
        wavelength_nm: CSV 记录的工作波长 (缺省 None)。
        exposure_ms: CSV 记录的曝光 (缺省 None)。
    """

    gray: np.ndarray
    y: np.ndarray
    slm_max_gray: int
    metric: str
    wavelength_nm: int | None = None
    exposure_ms: float | None = None


@dataclass
class SinFit:
    """灰度域一次谐波拟合参数 (与 gray_response.py 的模型同形)。

    Attributes:
        offset: 均值项 (> 0)。
        amplitude: 一次谐波幅度, 已归一为非负 (负幅度折进相位 +π)。
        period: 灰度域周期; 物理上应等于器件 2π 灰度数。
        phase: 相位, 已包裹到 [0, 2π)。
    """

    offset: float
    amplitude: float
    period: float
    phase: float

    @property
    def c(self) -> float:
        """孪生参数: 归一化强度一次谐波幅度。"""
        return self.amplitude / self.offset

    @property
    def psi_rad(self) -> float:
        """孪生参数: 灰度域相位偏移 ψ。"""
        return self.phase


def load_gray_csv(path: Path, metric: str) -> GrayResponse:
    """读取 gray_response.py 写出的 CSV (跳过其元数据注释行)。

    Args:
        path: CSV 路径。
        metric: ``sum`` 或 ``max``, 选择拟合哪一亮度列。

    Returns:
        解析好的 :class:`GrayResponse`。

    Raises:
        SystemExit: 文件缺失、列缺失、数值不可解析或样本太少。
    """
    if not path.is_file():
        sys.exit(f"错误: CSV 不存在: {path}")
    column = METRIC_COLUMNS[metric]
    text = path.read_text(encoding="utf-8")
    # gray_response.py 在表头前写了一行 "# SLM Serial: ..." 元数据注释
    body = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    if not body:
        sys.exit(f"错误: CSV 为空: {path}")
    reader = csv.DictReader(body)
    fieldnames = set(reader.fieldnames or [])
    if column not in fieldnames:
        # 旧的 gray_response CSV 没有 0 级桶列（只有全帧 sum / 峰值）。
        # 那就退回全帧总能量, 并说清楚 —— 静默换列会让人以为拟合的是桶能量。
        fallback = next(
            (c for c in ("sum_brightness", "max_brightness") if c in fieldnames),
            None,
        )
        if fallback is None:
            sys.exit(
                f"错误: CSV 缺少列 {column!r}, 也没有可退回的 "
                f"sum_brightness / max_brightness; 实际列: {reader.fieldnames}"
            )
        print(
            f"警告: CSV 没有 {column!r} 列（0 级桶能量），退回 {fallback!r}"
            f"（全帧能量，含杂散光）",
            file=sys.stderr,
        )
        column = fallback
    missing = {"gray_value", column, "slm_max_gray"} - fieldnames
    if missing:
        sys.exit(
            f"错误: CSV 缺少列 {sorted(missing)}; 实际列: {reader.fieldnames}"
        )
    gray: list[float] = []
    y: list[float] = []
    max_grays: set[int] = set()
    wavelengths: set[int] = set()
    exposures: set[float] = set()
    for row_no, row in enumerate(reader, start=1):
        try:
            gray.append(float(row["gray_value"]))
            y.append(float(row[column]))
            max_grays.add(int(row["slm_max_gray"]))
        except (TypeError, ValueError) as exc:
            sys.exit(f"错误: CSV 第 {row_no} 行数值不可解析: {exc}")
        if row.get("wavelength_nm", "").strip().isdigit():
            wavelengths.add(int(row["wavelength_nm"]))
        try:
            exposures.add(float(row.get("exposure_ms", "")))
        except (KeyError, TypeError, ValueError):
            # exposure_ms 只是元数据，缺这一列不该让整条拟合失败
            pass
    gray_arr = np.asarray(gray, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(gray_arr) < 4:
        sys.exit(f"错误: 可用数据点只有 {len(gray_arr)} 个 (< 4), 无法拟合 4 参数正弦")
    if not np.all(np.isfinite(y_arr)):
        sys.exit("错误: 亮度列含非有限值 (NaN/inf)")
    if len(set(gray_arr.tolist())) < 4:
        sys.exit("错误: 灰度列不足 4 个不同取值")
    if len(max_grays) != 1:
        print(f"警告: CSV 里 slm_max_gray 不一致 {sorted(max_grays)}, 取最大值")
    order = np.argsort(gray_arr)
    return GrayResponse(
        gray=gray_arr[order],
        y=y_arr[order],
        slm_max_gray=max(max_grays),
        metric=column,
        wavelength_nm=next(iter(wavelengths)) if len(wavelengths) == 1 else None,
        exposure_ms=next(iter(exposures)) if len(exposures) == 1 else None,
    )


def _sin_model(
    x: np.ndarray, offset: float, amplitude: float, period: float, phase: float
) -> np.ndarray:
    """一次谐波模型 (与 gray_response.py 的 ``_sin_model`` 完全同形)。"""
    return offset + amplitude * np.sin(2 * np.pi * x / period + phase)


def fit_coupling(data: GrayResponse, fix_period: bool) -> SinFit:
    """对亮度-灰度曲线做正弦拟合, 并归一化到 (c, ψ) 约定的符号/包裹。

    初值策略与 ``gray_response.py._fit_and_plot`` 一致:
    ``offset=mean(y)``, ``amplitude=half_range``, ``period=slm_max_gray``, ``phase=0``。

    Args:
        data: 解析好的 CSV。
        fix_period: True 时把周期钉在 CSV 的 ``slm_max_gray`` (退化为 3 参数拟合);
            False (默认) 时周期自由 —— 器件的 2π 灰度数是查询值而非常量。

    Returns:
        :class:`SinFit`; ``c``/``psi_rad`` 即孪生两参数。

    Raises:
        SystemExit: curve_fit 失败或 offset <= 0 (不会退回 c = 0)。
    """
    x, y = data.gray, data.y
    period0 = float(data.slm_max_gray)
    p0 = [float(y.mean()), float(y.max() - y.min()) / 2.0, period0, 0.0]
    try:
        if fix_period:
            def _model(
                xx: np.ndarray, offset: float, amplitude: float, phase: float
            ) -> np.ndarray:
                return _sin_model(xx, offset, amplitude, period0, phase)

            popt, _ = curve_fit(_model, x, y, p0=p0[:2] + p0[3:], maxfev=20000)
            fit = SinFit(float(popt[0]), float(popt[1]), period0, float(popt[2]))
        else:
            popt, _ = curve_fit(_sin_model, x, y, p0=p0, maxfev=20000)
            fit = SinFit(*(float(v) for v in popt))
    except Exception as exc:  # noqa: BLE001 - curve_fit 抛型杂, 统一转成退出
        sys.exit(f"错误: 正弦拟合失败: {exc} —— 不退回 c = 0, 请检查 CSV 质量")
    if fit.offset <= 0.0:
        sys.exit(
            f"错误: 拟合 offset = {fit.offset:.4g} <= 0, 亮度基线不可信, 拒绝换算 c"
        )
    if fit.amplitude < 0.0:  # 负幅度折进相位: -A·sin(θ) = A·sin(θ+π)
        fit.amplitude = -fit.amplitude
        fit.phase += math.pi
    fit.period = abs(fit.period)
    fit.phase = math.fmod(fit.phase, 2.0 * math.pi)
    if fit.phase < 0.0:
        fit.phase += 2.0 * math.pi
    return fit


def rel_spread(values: np.ndarray) -> float:
    """仓库调制检测口径: ``(max - min) / mean``。"""
    mean = float(np.mean(values))
    if mean == 0.0:
        return float("nan")
    return (float(np.max(values)) - float(np.min(values))) / mean


def base_config_dict(base_path: Path | None) -> dict:
    """给出要合并耦合参数的孪生配置字典 (缺省 = 全默认 BenchTwinConfig)。

    Args:
        base_path: 已有的孪生配置 JSON (``save_fit`` 形状或裸 config); None → 默认。

    Returns:
        可被 ``BenchTwin.from_dict`` 校验通过的配置字典 (深拷贝)。

    Raises:
        SystemExit: 基础配置本身不可加载。
    """
    if base_path is None:
        return BenchTwin(BenchTwinConfig()).to_dict()
    if not base_path.is_file():
        sys.exit(f"错误: --base-config 不存在: {base_path}")
    try:
        payload = json.loads(base_path.read_text(encoding="utf-8"))
        cfg_dict = payload.get("config", payload)
        BenchTwin.from_dict(cfg_dict)  # 先校验基础配置合法
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        sys.exit(f"错误: --base-config 不是合法孪生配置: {exc}")
    return json.loads(json.dumps(cfg_dict))


def build_output_config(
    base_path: Path | None, levels: int, fit: SinFit
) -> dict:
    """把拟合出的 (c, ψ) 与 gray_levels 写进孪生配置字典。

    Args:
        base_path: 可选基础配置。
        levels: 器件 2π 灰度数, 写入 ``errors.gray_levels``。
        fit: 拟合结果。

    Returns:
        校验通过的 config 字典。

    Raises:
        SystemExit: 合并后的配置过不了 ``BenchTwin.from_dict`` 校验 (例如
            基础配置带 ``lut_table`` 而新 ``gray_levels`` 与其长度不符)。
    """
    cfg = base_config_dict(base_path)
    errors = cfg.setdefault("errors", {})
    old_levels = int(errors.get("gray_levels", 1023))
    errors["gray_levels"] = int(levels)
    errors["slm_amplitude_coupling"] = float(fit.c)
    errors["slm_amplitude_coupling_phase_rad"] = float(fit.psi_rad)
    try:
        BenchTwin.from_dict(cfg)  # 再校验合并结果
    except (ValueError, KeyError) as exc:
        errors["gray_levels"] = old_levels
        sys.exit(
            f"错误: 合并 gray_levels={levels} 后孪生配置校验失败 (基础配置可能带 "
            f"lut_table, 其长度必须 = gray_levels + 1): {exc}"
        )
    return cfg


def check_config(data: GrayResponse, config_path: Path,
                 resid_tol: float, spread_tol: float) -> bool:
    """把已有配置里的 (c, ψ, levels) 拿去和 CSV 对比, 报告预测 vs 实测。

    预测曲线按最小二乘只做一次仿射配平 ``y ≈ a + b·I(g)`` (相机绝对标度未知,
    但形状/相对起伏完全由配置决定), 然后比较残差与 rel_spread。

    Args:
        data: 实测 CSV。
        config_path: 待校验的孪生配置 JSON。
        resid_tol: 归一化 RMS 残差 (rms / mean(y)) 容差。
        spread_tol: |rel_spread(实测) − rel_spread(预测)| 容差。

    Returns:
        True = 一致; False = 超出容差 (main 里映射为退出码 2)。
    """
    if not config_path.is_file():
        sys.exit(f"错误: --check 配置不存在: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        twin = BenchTwin.from_dict(payload.get("config", payload))
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        sys.exit(f"错误: --check 配置不是合法孪生配置: {exc}")
    c = float(twin.errors.slm_amplitude_coupling)
    psi = float(twin.errors.slm_amplitude_coupling_phase_rad)
    levels = int(twin.errors.gray_levels)
    x, y = data.gray, data.y
    intensity = 1.0 + c * np.sin(2.0 * math.pi * x / float(levels) + psi)
    design = np.column_stack([np.ones_like(x), intensity])
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = a + b * intensity
    residual = y - predicted
    mean_y = float(np.mean(y))
    rms = float(np.sqrt(np.mean(residual**2)))
    norm_rms = rms / mean_y if mean_y != 0.0 else float("inf")
    spread_meas = rel_spread(y)
    spread_pred = rel_spread(intensity)
    print(f"校验配置: {config_path}")
    print(f"  配置写的耦合: c = {c:.6f}, ψ = {psi:.6f} rad, gray_levels = {levels}")
    print(f"  仿射配平: y ≈ {a:.6g} + {b:.6g}·I(g)")
    print(f"  残差: max|res| = {np.max(np.abs(residual)):.6g}, "
          f"mean|res| = {np.mean(np.abs(residual)):.6g}, "
          f"RMS/mean(y) = {norm_rms:.4%}")
    print(f"  rel_spread: 实测 {spread_meas:.4f} vs 预测 {spread_pred:.4f} "
          f"(差 {abs(spread_meas - spread_pred):.4f})")
    print(f"  容差: RMS/mean ≤ {resid_tol:.2%}, |Δrel_spread| ≤ {spread_tol}")
    ok = norm_rms <= resid_tol and abs(spread_meas - spread_pred) <= spread_tol
    if not ok:
        print("结论: 配置的 (c, ψ) 与该 CSV **不一致** (退出码 2)")
    else:
        print("结论: 配置与 CSV 一致 (在容差内)")
    return ok


def make_plot(data: GrayResponse, fit: SinFit, out: Path) -> None:
    """画实测点 + 拟合曲线 (matplotlib Agg; 图内文字一律英文, 容器无 CJK 字体)。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = data.gray, data.y
    x_smooth = np.linspace(float(x.min()), float(x.max()), 500)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.scatter(x, y, s=30, color="royalblue", zorder=3, label="Measured")
    ax.plot(x_smooth, _sin_model(x_smooth, fit.offset, fit.amplitude,
                                 fit.period, fit.phase),
            color="crimson", lw=2, zorder=2,
            label=(f"fit: period={fit.period:.1f}, c={fit.c:.4f}, "
                   f"psi={fit.psi_rad:.3f} rad"))
    ax.axvline(float(data.slm_max_gray), color="gray", ls=":", lw=1,
               label=f"full-scale gray = {data.slm_max_gray}")
    ax.set_xlabel("SLM gray code")
    ax.set_ylabel(f"{data.metric} (ADU)")
    ax.set_title("SLM gray-to-amplitude coupling fit")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"图已保存: {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description="从灰度响应 CSV 拟合 SLM 灰度→振幅耦合 (c, ψ), 写出孪生配置 JSON。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv", type=Path, required=True,
                   help="gray_response.py 写出的灰度响应 CSV")
    p.add_argument(
        "--metric",
        choices=("sum", "max", "bucket"),
        default="bucket",
        help="拟合的亮度列: bucket = 冻结 0 级桶能量 (**物理正确的耦合观测量**, "
             "平场相位的 0 阶就是瞳面场的直流分量); sum = 全帧总能量; "
             "max = 峰值 (对光斑形状漂移更敏感)",
    )
    p.add_argument("--fix-period", action="store_true",
                   help="把灰度域周期钉在 CSV 的 slm_max_gray (默认自由拟合: "
                        "器件 2π 灰度数是查询值而非常量)")
    p.add_argument("--levels", type=int, default=DEFAULT_LEVELS,
                   help=f"孪生的 2π 灰度数 (1064 nm 下 993; 不是 10-bit 满量程 "
                        f"1023), 默认 {DEFAULT_LEVELS}")
    p.add_argument("--base-config", type=Path, default=None,
                   help="可选: 已有孪生配置 JSON (save_fit 形状), 在其上合并耦合参数")
    p.add_argument("--out", type=Path, default=None,
                   help="输出 JSON (默认 data/debug/slm_amplitude_coupling/<ts>/"
                        "twin_config.json)")
    p.add_argument("--check", type=Path, default=None,
                   help="只校验: 把该配置里写的 c/ψ 拿去与 --csv 对比, 报告残差")
    p.add_argument("--plot", type=Path, default=None,
                   help="可选: 画出实测点与拟合曲线 (PNG, 图内文字用英文)")
    p.add_argument("--resid-tol", type=float, default=0.05,
                   help="--check 容差: 归一化 RMS 残差 rms/mean(y) 上限 (默认 0.05)")
    p.add_argument("--spread-tol", type=float, default=0.02,
                   help="--check 容差: |Δrel_spread| 上限 (默认 0.02)")
    args = p.parse_args(argv)
    if args.levels < 1:
        p.error("--levels 必须 >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    """入口: 拟合 → 换算 (c, ψ) → 写孪生配置; 或 ``--check`` 只校验。"""
    args = parse_args(argv)
    data = load_gray_csv(args.csv, args.metric)

    print(f"CSV: {args.csv}  拟合列: {data.metric}  数据点: {len(data.gray)}")
    if data.wavelength_nm is not None:
        print(f"  CSV 记录: 波长 {data.wavelength_nm} nm, 曝光 "
              f"{data.exposure_ms} ms, slm_max_gray = {data.slm_max_gray}")
    if args.levels == data.slm_max_gray:
        print(f"警告: --levels = {args.levels} 与 CSV 的 slm_max_gray 相同 —— "
              f"1023 是 10-bit 满量程, 不是 2π 灰度数 (1064 nm 应为 993); "
              f"若这不是本意, 请核对 --levels")
    elif args.levels != data.slm_max_gray:
        print(f"提示: --levels = {args.levels} ≠ CSV 的 slm_max_gray = "
              f"{data.slm_max_gray} (满量程)。这是**预期**区分: 前者是 2π 灰度数 "
              f"(孪生的灰度域周期), 后者只是扫描上界。")

    if args.check is not None:
        return 0 if check_config(data, args.check, args.resid_tol,
                                 args.spread_tol) else 2

    fit = fit_coupling(data, fix_period=args.fix_period)
    spread_meas = rel_spread(data.y)
    fitted_curve = _sin_model(data.gray, fit.offset, fit.amplitude,
                              fit.period, fit.phase)
    spread_fit = rel_spread(fitted_curve)

    print("\n拟合 (模型 y = offset + amplitude·sin(2π·g/period + phase), "
          "与 gray_response.py 同形):")
    print(f"  offset   = {fit.offset:.6g}")
    print(f"  amplitude= {fit.amplitude:.6g}")
    print(f"  period   = {fit.period:.6g}"
          f"{' (钉在 slm_max_gray)' if args.fix_period else ' (自由拟合)'}")
    print(f"  phase    = {fit.phase:.6f} rad")
    print("换算到孪生 I(g) = 1 + c·sin(2π·g/levels + ψ):")
    print(f"  c   = amplitude/offset = {fit.c:.6f}")
    print(f"  ψ   = phase            = {fit.psi_rad:.6f} rad")
    print(f"  levels (写入 gray_levels) = {args.levels}")
    if not 0.0 <= fit.c < 1.0:
        sys.exit(f"错误: 拟合 c = {fit.c:.6f} 不在孪生有意义范围 [0, 1), 结果不可写入配置")
    if fit.c == 0.0:
        print("警告: 拟合 c = 0 —— 该 CSV 分辨不出耦合 (可能光太弱或数据太短), "
              "写入的配置等价于关闭耦合项")

    period_dev = abs(fit.period - args.levels) / args.levels
    if period_dev > PERIOD_LEVELS_TOL:
        print(f"\n⚠️  警告 (大声): 拟合周期 {fit.period:.1f} 与孪生 levels "
              f"{args.levels} 相差 {period_dev:.1%} > {PERIOD_LEVELS_TOL:.0%}。"
              f"孪生模型假设灰度域谐波周期 **恰好等于** gray_levels "
              f"(I(g) = 1 + c·sin(2π·g/levels + ψ)); 若真实 2π 灰度数是 "
              f"{fit.period:.1f}, 请把 --levels 设成器件查询值, 否则 ψ 与 c 的"
              f"灰度域位置会错位。")

    print(f"\n判据对照 (仓库调制检测: rel_spread > {VERDICT_REL_SPREAD} ⇒ c ≳ 0.075):")
    print(f"  实测 rel_spread   = {spread_meas:.4f}")
    print(f"  拟合曲线 rel_spread = {spread_fit:.4f} (= 2c 在覆盖整周期时)")
    print(f"  拟合蕴含 c        = {fit.c:.4f} → "
          f"{'通过' if spread_fit > VERDICT_REL_SPREAD else '不通过'} "
          f"(rel_spread {'>' if spread_fit > VERDICT_REL_SPREAD else '≤'} "
          f"{VERDICT_REL_SPREAD})")

    if args.plot is not None:
        make_plot(data, fit, args.plot)

    config = build_output_config(args.base_config, args.levels, fit)
    out = args.out
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("data") / "debug" / "slm_amplitude_coupling" / ts / "twin_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "amplitude_coupling_fit": {
            "csv": str(args.csv),
            "metric": data.metric,
            "fit_model": "offset + amplitude*sin(2*pi*g/period + phase)",
            "offset": fit.offset,
            "amplitude": fit.amplitude,
            "period": fit.period,
            "phase_rad": fit.phase,
            "c": fit.c,
            "psi_rad": fit.psi_rad,
            "levels": int(args.levels),
            "fix_period": bool(args.fix_period),
            "rel_spread_measured": spread_meas,
            "rel_spread_fitted": spread_fit,
            "verdict_rel_spread_threshold": VERDICT_REL_SPREAD,
            "slm_max_gray": int(data.slm_max_gray),
            "wavelength_nm": data.wavelength_nm,
            "exposure_ms": data.exposure_ms,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n孪生配置已写出: {out}")
    print("可直接传给: scripts/slm/gsnet_transfer.py --twin-config 与 "
          "scripts/slm/export_gsnet_phase.py --twin-config (load_fit 可读回)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
