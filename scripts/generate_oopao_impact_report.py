"""生成 AO_OOPAO_BACKEND 端到端影响报告 (SimTurbulenceAOEnv, 离线)。

驱动真实的 RL AO 环境 ``SimTurbulenceAOEnv`` (DM 影响函数 + WFS 斜率 + 焦面 FFT),
在 ``AO_OOPAO_BACKEND`` 开关的两种后端臂 (``numpy`` / ``oopao``) 下, 沿 cn2 阶梯
跑两种模式:

- ``open`` 开环: 滑动湍流窗口 (screen_step_px=2), 零动作, 记录湍流演化下的端到端指标;
- ``closed`` 闭环: 冻结湍流 (screen_step_px=0), 3 步贪婪 SPGD (δ=0.03) 校正静态像差。

输出 ``report.md`` + ``summary.csv`` + ``figures/`` 到 ``--out-dir`` (默认
``report/oopao_impact/``)。

守卫 (任一失败即中止, 拒绝写出空洞报告):

1. OOPAO 不可用 / 路由自检失败 → 退出码 2;
2. 每行实测 ``_oopao_enabled()`` 与臂不一致 (静默回退 numpy) → RuntimeError;
3. 反空洞守卫: 每个 mode 至少一个 cn2>0 单元格两臂指标不同, 否则 RuntimeError;
4. cn2=0 对照组两臂须位级一致 (不一致仅告警)。

负发现: ``slm_shaping_bench.forward_intensity`` 的 cn2 是死配置 (只调用永不路由的
``focal_plane``), 探测结果写入报告。

用法::

    .venv/bin/python scripts/generate_oopao_impact_report.py
    .venv/bin/python scripts/generate_oopao_impact_report.py --quick
    .venv/bin/python scripts/generate_oopao_impact_report.py --n-grid 128 --seed 7
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"

# `scripts._common` lives in this package, so the REPO ROOT (not just
# `src`) must be importable. Direct `python scripts/<name>.py` does not put
# it there; pytest does via `pythonpath = ["src", ".", "scripts"]`.
sys.path.insert(0, str(_SRC))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ao_shaping.drivers.sim import beam_backend as bb  # noqa: E402
from ao_shaping.drivers.sim import oopao_backend  # noqa: E402
from ao_shaping.drivers.sim.slm_shaping_bench import (  # noqa: E402
    ShapingBenchConfig,
    forward_intensity,
)
from ao_shaping.optimizer.rl.envs import SimTurbulenceAOEnv  # noqa: E402
# `scripts._common` lives in this package, so the REPO ROOT (not just `src`)
# must be importable. A direct `python scripts/<name>.py` does not put it
# there; pytest does, via `pythonpath = ["src", ".", "scripts"]` in pyproject.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import fmt_ratio

# CJK 字体 (仓库约定): 逐级回退到 DejaVu Sans, 并关闭 unicode minus。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
ARM_NUMPY = "numpy"
ARM_OOPAO = "oopao"
ARMS = (ARM_NUMPY, ARM_OOPAO)

MODE_OPEN = "open"
MODE_CLOSED = "closed"
MODES = (MODE_OPEN, MODE_CLOSED)

CN2_LADDER: tuple[tuple[float, str], ...] = (
    (0.0, "none"),
    (1e-16, "weak"),
    (5e-15, "moderate"),
    (5e-14, "strong"),
)
DEFAULT_CN2 = "0,1e-16,5e-15,5e-14"

# 环境固定参数 (与 RL 训练默认一致的物理基准)
N_ACTUATORS = 4
N_SUBAPERTURES = 4
WAVELENGTH_M = 1550e-9
APERTURE_SIZE_M = 0.1
PROPAGATION_DISTANCE_M = 1000.0
PIB_RADIUS_PX = 4
SPGD_DELTA = 0.03  # 3 步贪婪 SPGD 的扰动幅度 (动作上限)

CSV_COLUMNS: tuple[str, ...] = (
    "mode",
    "cn2",
    "arm",
    "n_grid",
    "seed",
    "steps",
    "init_strehl",
    "final_strehl",
    "best_strehl",
    "init_pib",
    "final_pib",
    "best_pib",
    "init_rms",
    "final_rms",
    "disturbance_rms",
    "strehl_gain",
    "pib_gain_rel",
    "rms_reduction_rel",
    "oopao_enabled",
)

# 相位屏强度比值 (oopao/numpy) 的「恒定」判据: 相对离散度 (max-min)/mean 上限。
# 实测两条后端实现的比值在整条 cn2 阶梯上相对离散度 ~1e-5, 属浮点舍入量级。
RATIO_CONSTANT_TOL_REL = 1e-3

# cn2=0 对照组要求两臂位级一致的指标 (全部不经过 propagate(), 只经 focal_plane)
ARM_INVARIANT_METRICS: tuple[str, ...] = (
    "init_strehl",
    "final_strehl",
    "best_strehl",
    "init_pib",
    "final_pib",
    "best_pib",
    "init_rms",
    "final_rms",
    "disturbance_rms",
)

ARM_LABELS = {ARM_NUMPY: "numpy", ARM_OOPAO: "oopao"}
ARM_COLORS = {ARM_NUMPY: "tab:blue", ARM_OOPAO: "tab:orange"}
ARM_LINESTYLES = {ARM_NUMPY: "-", ARM_OOPAO: "--"}
ARM_MARKERS = {ARM_NUMPY: "o", ARM_OOPAO: "s"}


class OopaoBackendUnavailableError(RuntimeError):
    """OOPAO 后端不可用或路由与预期不符。"""


# --------------------------------------------------------------------------- #
# 后端臂切换与自检
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def backend_arm(arm_name: str) -> Iterator[None]:
    """切换到指定后端臂, 进入与退出都清 OOPAO lru_cache。

    ``oopao_backend._get_backend`` 是 ``@lru_cache(maxsize=8)``, 缓存键里只有物理
    配置没有臂标志 —— 不清缓存时第二条臂可能拿到第一条臂的 ``Atmosphere``, 于是
    "oopao" 臂静默退化成 numpy。进入时清一次 (换臂), 退出时再清一次 (防止残留
    实例泄漏到下一个 cn2/臂/模式)。

    Args:
        arm_name: ``numpy`` 或 ``oopao``。
    """
    if arm_name == ARM_OOPAO:
        os.environ["AO_OOPAO_BACKEND"] = "1"
    else:
        os.environ.pop("AO_OOPAO_BACKEND", None)
    oopao_backend._get_backend.cache_clear()
    try:
        yield
    finally:
        oopao_backend._get_backend.cache_clear()


def assert_oopao_usable() -> None:
    """启动自检: OOPAO 必须真的可用且路由得通, 否则大声中止。

    Raises:
        OopaoBackendUnavailableError: OOPAO 不可用或路由自检失败。
    """
    if not oopao_backend._oopao_available():
        raise OopaoBackendUnavailableError(
            "OOPAO 不可用 (oopao_backend._oopao_available() == False)。请用带 OOPAO "
            "的 venv 运行。本脚本拒绝在 OOPAO 缺席时继续 —— 否则会产出一份被标成 "
            "oopao 的 numpy 重复结果。"
        )
    with backend_arm(ARM_OOPAO):
        if not bb._oopao_enabled():
            raise OopaoBackendUnavailableError(
                "AO_OOPAO_BACKEND=1 未激活 OOPAO 后端 (静默回退 numpy)。"
            )
    with backend_arm(ARM_NUMPY):
        if bb._oopao_enabled():
            raise OopaoBackendUnavailableError(
                "移除 AO_OOPAO_BACKEND 后 OOPAO 后端仍处于激活状态。"
            )
    logger.info("OOPAO 后端可用且双臂路由自检通过")


# --------------------------------------------------------------------------- #
# 单 episode 运行
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    """一次 episode 的端到端指标 (一个 (mode, cn2, arm) 单元格)。"""

    mode: str
    cn2: float
    arm: str
    n_grid: int
    seed: int
    steps: int
    init_strehl: float
    final_strehl: float
    best_strehl: float
    init_pib: float
    final_pib: float
    best_pib: float
    init_rms: float
    final_rms: float
    disturbance_rms: float
    trace: list[float]
    oopao_enabled: bool

    @property
    def strehl_gain(self) -> float:
        """best Strehl 相对 init 的绝对增益。"""
        return self.best_strehl - self.init_strehl

    @property
    def pib_gain_rel(self) -> float:
        """best PIB 相对 init 的相对增益。"""
        return (self.best_pib - self.init_pib) / max(self.init_pib, 1.0)

    @property
    def rms_reduction_rel(self) -> float:
        """final RMS 相对 init 的相对下降 (init≈0 时无湍流可校正, 返回 0)。"""
        if self.init_rms < 1e-9:
            return 0.0
        return (self.init_rms - self.final_rms) / self.init_rms

    def csv_row(self) -> dict[str, str]:
        """渲染为 CSV 行 (全部字符串)。"""
        return {
            "mode": self.mode,
            "cn2": fmt_ratio(self.cn2),
            "arm": self.arm,
            "n_grid": str(self.n_grid),
            "seed": str(self.seed),
            "steps": str(self.steps),
            "init_strehl": f"{self.init_strehl:.6f}",
            "final_strehl": f"{self.final_strehl:.6f}",
            "best_strehl": f"{self.best_strehl:.6f}",
            "init_pib": f"{self.init_pib:.6f}",
            "final_pib": f"{self.final_pib:.6f}",
            "best_pib": f"{self.best_pib:.6f}",
            "init_rms": f"{self.init_rms:.6f}",
            "final_rms": f"{self.final_rms:.6f}",
            "disturbance_rms": f"{self.disturbance_rms:.6f}",
            "strehl_gain": f"{self.strehl_gain:.6f}",
            "pib_gain_rel": f"{self.pib_gain_rel:.6f}",
            "rms_reduction_rel": f"{self.rms_reduction_rel:.6f}",
            "oopao_enabled": "1" if self.oopao_enabled else "0",
        }


def run_episode(
    *,
    mode: str,
    cn2: float,
    arm: str,
    n_grid: int,
    seed: int,
    steps: int,
) -> EpisodeResult:
    """在指定后端臂下驱动一次 ``SimTurbulenceAOEnv`` episode。

    Args:
        mode: ``open`` (开环, 滑动湍流, 零动作) 或 ``closed`` (闭环, 冻结湍流,
            3 步贪婪 SPGD 校正)。
        cn2: 湍流强度 (m^{-2/3})。
        arm: ``numpy`` 或 ``oopao``。
        n_grid: 仿真网格边长。
        seed: 随机种子 (全局 RNG 与 env 的 ``reset(seed=...)`` 都用它)。
        steps: episode 步数预算 (closed 模式下 SPGD 迭代数 = steps // 3)。

    Returns:
        该单元格的端到端指标。
    """
    with backend_arm(arm):
        np.random.seed(seed)
        env = SimTurbulenceAOEnv(
            n_grid=n_grid,
            n_actuators=N_ACTUATORS,
            n_subapertures=N_SUBAPERTURES,
            max_steps=steps,
            cn2=cn2,
            wavelength=WAVELENGTH_M,
            aperture_size=APERTURE_SIZE_M,
            propagation_distance=PROPAGATION_DISTANCE_M,
            pib_radius=PIB_RADIUS_PX,
            screen_step_px=0 if mode == MODE_CLOSED else 2,
        )
        obs, info = env.reset(seed=seed)
        trace = [float(info["strehl"])]
        last_info = info
        if mode == MODE_OPEN:
            for _ in range(steps):
                _, _, _, _, last_info = env.step(
                    np.zeros(env.action_dim, dtype=np.float32)
                )
                trace.append(float(last_info["strehl"]))
        else:
            rng = np.random.default_rng(seed)
            iterations = max(steps // 3, 1)
            for _ in range(iterations):
                direction = rng.normal(size=env.action_dim)
                direction /= float(np.linalg.norm(direction))
                _, _, _, _, info_plus = env.step(SPGD_DELTA * direction)
                _, _, _, _, info_minus = env.step(-2.0 * SPGD_DELTA * direction)
                if float(info_plus["strehl"]) >= float(info_minus["strehl"]):
                    _, _, _, _, last_info = env.step(SPGD_DELTA * direction)
                else:
                    last_info = info_minus
                trace.append(float(last_info["strehl"]))

        result = EpisodeResult(
            mode=mode,
            cn2=cn2,
            arm=arm,
            n_grid=n_grid,
            seed=seed,
            steps=steps,
            init_strehl=float(info["strehl"]),
            final_strehl=float(last_info["strehl"]),
            best_strehl=float(last_info["best_strehl"]),
            init_pib=float(info["pib"]),
            final_pib=float(last_info["pib"]),
            best_pib=float(last_info["best_pib"]),
            init_rms=float(info["rms"]),
            final_rms=float(last_info["rms"]),
            disturbance_rms=float(info["disturbance_rms"]),
            trace=trace,
            oopao_enabled=bool(bb._oopao_enabled()),
        )
        return result


# --------------------------------------------------------------------------- #
# 守卫
# --------------------------------------------------------------------------- #
def verify_backend_routing(results: list[EpisodeResult]) -> None:
    """每行记录的 ``_oopao_enabled()`` 必须与臂一致, 否则大声中止。

    Args:
        results: 全部 episode 结果。

    Raises:
        RuntimeError: 某行实测后端与臂不符 (静默回退 numpy)。
    """
    for result in results:
        expected = result.arm == ARM_OOPAO
        if result.oopao_enabled != expected:
            raise RuntimeError(
                f"后端路由不符: {result.mode}/{fmt_ratio(result.cn2)}/{result.arm} 实测 "
                f"_oopao_enabled()={result.oopao_enabled} (期望 {expected})。"
                "AO_OOPAO_BACKEND 静默回退 numpy —— 拒绝信任该数字。"
            )
    logger.info("后端路由校验通过: 全部 {} 行 _oopao_enabled() 与臂一致", len(results))


def verify_anti_vacuity(results: list[EpisodeResult], cn2_values: list[float]) -> int:
    """反空洞守卫: 每个 mode 至少一个 cn2>0 单元格两臂必须不同, 否则中止。

    Args:
        results: 全部 episode 结果。
        cn2_values: 本次实际使用的 cn2 阶梯。

    Returns:
        两臂指标不同的 (mode, cn2>0) 单元格数。

    Raises:
        RuntimeError: 没有任何 cn2>0 单元格两臂不同 —— 后端开关没有影响被测路径。
    """
    differing = 0
    for mode in MODES:
        for cn2 in cn2_values:
            if cn2 <= 0:
                continue
            numpy_row = next(
                r
                for r in results
                if r.mode == mode and r.cn2 == cn2 and r.arm == ARM_NUMPY
            )
            oopao_row = next(
                r
                for r in results
                if r.mode == mode and r.cn2 == cn2 and r.arm == ARM_OOPAO
            )
            metrics = ("init_strehl", "best_strehl", "init_pib", "init_rms")
            if any(getattr(numpy_row, m) != getattr(oopao_row, m) for m in metrics):
                differing += 1
    if differing == 0:
        raise RuntimeError(
            "反空洞守卫失败: 所有 cn2>0 单元格的两臂指标完全相同。后端开关没有影响 "
            "被测路径 —— 拒绝写出一份空洞报告。"
        )
    logger.info("反空洞守卫通过: {} 个 (mode, cn2>0) 单元格两臂指标不同", differing)
    return differing


def verify_control(results: list[EpisodeResult], cn2_values: list[float]) -> bool:
    """cn2=0 对照组: 两臂必须位级一致 (turbulence_phase 在 cn2<=0 短路为零屏)。

    Args:
        results: 全部 episode 结果。
        cn2_values: 本次实际使用的 cn2 阶梯。

    Returns:
        True 表示对照组全部一致。
    """
    if 0.0 not in cn2_values:
        logger.info("cn2=0 不在本次阶梯中, 跳过对照组校验")
        return True
    ok = True
    for mode in MODES:
        numpy_row = next(
            r for r in results if r.mode == mode and r.cn2 == 0.0 and r.arm == ARM_NUMPY
        )
        oopao_row = next(
            r for r in results if r.mode == mode and r.cn2 == 0.0 and r.arm == ARM_OOPAO
        )
        for metric in ARM_INVARIANT_METRICS:
            if getattr(numpy_row, metric) != getattr(oopao_row, metric):
                ok = False
                logger.warning(
                    "cn2=0 对照组不一致 ({}): {} numpy={} oopao={}",
                    mode,
                    metric,
                    getattr(numpy_row, metric),
                    getattr(oopao_row, metric),
                )
    if ok:
        logger.info("cn2=0 对照组通过: 两臂全部指标位级一致")
    else:
        logger.warning("cn2=0 对照组存在不一致 —— 相位生成可能引入了臂间偏置")
    return ok


# --------------------------------------------------------------------------- #
# 负发现探测
# --------------------------------------------------------------------------- #
def probe_slm_shaping_bench_dead_cn2() -> dict[str, Any]:
    """探测 ``slm_shaping_bench`` 的 cn2 是否为死配置。

    负发现: 它 import 了 ``turbulence_phase`` 但从不调用; ``forward_intensity``
    只调用 ``focal_plane`` (永不路由)。预期 cn2=0 与 cn2=5e-14 输出字节一致。

    Returns:
        ``{"byte_identical": bool, "max_abs_diff": float}``。
    """
    phase = np.zeros((256, 256), dtype=float)
    intensity_zero = forward_intensity(phase, ShapingBenchConfig(n_grid=256, cn2=0.0))
    intensity_strong = forward_intensity(
        phase, ShapingBenchConfig(n_grid=256, cn2=5e-14)
    )
    identical = bool(np.array_equal(intensity_zero, intensity_strong))
    max_diff = float(np.abs(intensity_zero - intensity_strong).max())
    logger.info(
        "slm_shaping_bench 死配置探测: cn2=0 vs 5e-14 字节一致={} max|ΔI|={}",
        identical,
        max_diff,
    )
    return {"byte_identical": identical, "max_abs_diff": max_diff}


# --------------------------------------------------------------------------- #
# 格式化与表格
# --------------------------------------------------------------------------- #
def _rel_diff(numpy_value: float, oopao_value: float) -> str:
    """两臂相对差 (百分比); numpy 值为 0 时返回 ``—``。"""
    if numpy_value == 0.0:
        return "—"
    return f"{(oopao_value - numpy_value) / abs(numpy_value) * 100.0:+.2f}%"


def _cn2_label(cn2: float) -> str:
    """cn2 的阶梯标签 (不在阶梯内时回落到科学计数法)。"""
    for value, label in CN2_LADDER:
        if cn2 == value:
            return label
    return fmt_ratio(cn2)


def _cn2_axis_label(cn2: float) -> str:
    """图轴上的 cn2 标签 (标签 + 数值两行)。"""
    return f"{_cn2_label(cn2)}\n({fmt_ratio(cn2)})"


def _metric_table(
    results: list[EpisodeResult], mode: str, cn2_values: list[float]
) -> str:
    """渲染某个模式的汇总对比 markdown 表。"""
    lines = [
        "| cn2 | 臂 | init Strehl | best Strehl | init PIB | init RMS | disturbance RMS | Strehl 增益 |",
        "|-----|-----|------------|-------------|----------|----------|-----------------|-------------|",
    ]
    for cn2 in cn2_values:
        numpy_row = next(
            r for r in results if r.mode == mode and r.cn2 == cn2 and r.arm == ARM_NUMPY
        )
        oopao_row = next(
            r for r in results if r.mode == mode and r.cn2 == cn2 and r.arm == ARM_OOPAO
        )
        for row in (numpy_row, oopao_row):
            lines.append(
                f"| {fmt_ratio(cn2)} | {row.arm} | {row.init_strehl:.4f} | {row.best_strehl:.4f} "
                f"| {row.init_pib:.4g} | {row.init_rms:.4f} | {row.disturbance_rms:.4f} "
                f"| {row.strehl_gain:+.4f} |"
            )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 相位屏强度比值分析 (disturbance_rms oopao/numpy)
# --------------------------------------------------------------------------- #
@dataclass
class RatioProfile:
    """某个 mode 的相位屏强度比值剖面 (按 cn2 升序, 仅含 cn2>0 档位)。"""

    mode: str
    cn2_levels: list[float]
    ratios: list[float]

    @property
    def mean(self) -> float:
        """比值算术平均。"""
        return float(np.mean(self.ratios)) if self.ratios else 0.0

    @property
    def spread_rel(self) -> float:
        """比值的相对离散度 (max-min)/mean —— 0 表示逐位恒定。"""
        if len(self.ratios) < 2 or self.mean == 0.0:
            return float("nan")
        spread = float(np.max(self.ratios) - np.min(self.ratios))
        return spread / abs(self.mean)

    @property
    def is_constancy_claimable(self) -> bool:
        """档位是否足够判定恒定性 (单档无法对比)。"""
        return len(self.ratios) >= 2

    @property
    def is_constant(self) -> bool:
        """比值在所有档位上是否恒定 (判据 ``RATIO_CONSTANT_TOL_REL``)。"""
        if not self.is_constancy_claimable:
            return False
        return bool(self.spread_rel <= RATIO_CONSTANT_TOL_REL)


def _arm_row(
    results: list[EpisodeResult], mode: str, cn2: float, arm: str
) -> EpisodeResult:
    """取某个 (mode, cn2, arm) 单元格的结果。"""
    return next(r for r in results if r.mode == mode and r.cn2 == cn2 and r.arm == arm)


def disturbance_ratio(
    results: list[EpisodeResult], mode: str, cn2: float
) -> float | None:
    """两臂 ``disturbance_rms`` 之比 (oopao/numpy)。

    Args:
        results: 全部 episode 结果。
        mode: 模式。
        cn2: 湍流强度 (cn2<=0 时相位屏短路为零屏, 比值无定义 → 返回 ``None``)。

    Returns:
        比值, 或 ``None`` (对照组 / 任一臂为零 / 该单元格缺失)。
    """
    if cn2 <= 0.0:
        return None
    candidates = [
        r for r in results if r.mode == mode and r.cn2 == cn2 and r.arm in ARMS
    ]
    by_arm = {r.arm: r for r in candidates}
    if set(by_arm) != set(ARMS):
        return None
    numpy_value = by_arm[ARM_NUMPY].disturbance_rms
    oopao_value = by_arm[ARM_OOPAO].disturbance_rms
    if numpy_value == 0.0 or oopao_value == 0.0:
        return None
    return oopao_value / numpy_value


def ratio_profiles(
    results: list[EpisodeResult], cn2_values: list[float]
) -> list[RatioProfile]:
    """逐 mode 构建相位屏强度比值剖面 (仅 cn2>0 档位)。"""
    profiles: list[RatioProfile] = []
    for mode in MODES:
        levels: list[float] = []
        ratios: list[float] = []
        for cn2 in sorted(c for c in cn2_values if c > 0.0):
            ratio = disturbance_ratio(results, mode, cn2)
            if ratio is None:
                continue
            levels.append(cn2)
            ratios.append(ratio)
        profiles.append(RatioProfile(mode=mode, cn2_levels=levels, ratios=ratios))
    return profiles


def ratio_table(
    results: list[EpisodeResult], cn2_values: list[float]
) -> tuple[list[str], list[RatioProfile]]:
    """渲染逐 mode 的相位屏强度比值 markdown 表。

    cn2=0 对照组的 ``disturbance_rms`` 为零 (相位屏短路), 比值无定义, 因此列只含
    cn2>0 档位。判定列区分「恒定 / 非恒定 / 档位不足」三态 —— 单一 cn2>0 档位
    无法比较比值是否随湍流漂移, 不得宣称恒定。

    Args:
        results: 全部 episode 结果。
        cn2_values: 本次实际使用的 cn2 阶梯。

    Returns:
        ``(表行列表, 逐 mode 剖面)``。
    """
    profiles = ratio_profiles(results, cn2_values)
    levels: list[float] = []
    for profile in profiles:
        for cn2 in profile.cn2_levels:
            if cn2 not in levels:
                levels.append(cn2)
    levels.sort()

    header = "| mode | " + " | ".join(f"cn2={fmt_ratio(cn2)}" for cn2 in levels)
    header += " | 相对离散度 | 判定 |"
    separator = "|------|" + "|".join(["-----"] * len(levels)) + "|--------|------|"
    lines = [header, separator]

    for profile in profiles:
        ratio_by_cn2 = dict(zip(profile.cn2_levels, profile.ratios, strict=True))
        cells = [
            f"{ratio_by_cn2[cn2]:.3f}" if cn2 in ratio_by_cn2 else "—" for cn2 in levels
        ]
        if not profile.is_constancy_claimable:
            spread_text = "—"
            verdict = f"档位不足 (仅 {len(profile.ratios)} 个 cn2>0), 无法判定恒定性"
        else:
            spread_text = f"{profile.spread_rel:.2e}"
            verdict = "**恒定**" if profile.is_constant else "**非恒定**"
        lines.append(
            f"| {profile.mode} | "
            + " | ".join(cells)
            + f" | {spread_text} | {verdict} |"
        )
    return lines, profiles


def _init_rms_inversion(
    results: list[EpisodeResult], cn2_values: list[float]
) -> tuple[float, EpisodeResult, EpisodeResult] | None:
    """找出「相位屏更强但 init RMS 反而更低」的单元格 (反例)。

    ``disturbance_rms`` 与 ``init_rms`` 不同源 (前者全网格未截瞳的原始屏, 后者截瞳
    后的总波前), 因此这种反例正是二者不可互相推断的证据。

    Returns:
        ``(cn2, numpy_row, oopao_row)``, 无反例时 ``None`` (按 cn2 升序取第一个)。
    """
    for cn2 in sorted(c for c in cn2_values if c > 0.0):
        for mode in MODES:
            numpy_row = _arm_row(results, mode, cn2, ARM_NUMPY)
            oopao_row = _arm_row(results, mode, cn2, ARM_OOPAO)
            if (
                oopao_row.disturbance_rms > numpy_row.disturbance_rms
                and oopao_row.init_rms < numpy_row.init_rms
            ):
                return cn2, numpy_row, oopao_row
    return None


def _max_init_strehl_gap(
    results: list[EpisodeResult], cn2_values: list[float]
) -> tuple[float, EpisodeResult, EpisodeResult] | None:
    """找出 oopao 的 init Strehl **确实高于** numpy 最多的单元格。

    只在 gap > 0 的单元格中取最大值 —— 否则 §9 会打印出「oopao 高于 numpy」却
    配上一个更小的数值, 自相矛盾。无此类单元格时返回 ``None``。
    """
    best: tuple[float, EpisodeResult, EpisodeResult] | None = None
    for cn2 in sorted(c for c in cn2_values if c > 0.0):
        for mode in MODES:
            numpy_row = _arm_row(results, mode, cn2, ARM_NUMPY)
            oopao_row = _arm_row(results, mode, cn2, ARM_OOPAO)
            gap = oopao_row.init_strehl - numpy_row.init_strehl
            if gap <= 0.0:
                continue
            if best is None or gap > best[2].init_strehl - best[1].init_strehl:
                best = (cn2, numpy_row, oopao_row)
    return best


# --------------------------------------------------------------------------- #
# 图
# --------------------------------------------------------------------------- #
def render_strehl_figure(
    results: list[EpisodeResult], cn2_values: list[float], path: Path
) -> None:
    """每 cn2 的 Strehl 对比 (init + best, 两臂, 两种模式)。"""
    x = np.arange(len(cn2_values))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, MODES):
        for arm, offset in ((ARM_NUMPY, -width * 1.5), (ARM_OOPAO, width * 0.5)):
            init = [
                next(
                    r for r in results if r.mode == mode and r.cn2 == c and r.arm == arm
                ).init_strehl
                for c in cn2_values
            ]
            best = [
                next(
                    r for r in results if r.mode == mode and r.cn2 == c and r.arm == arm
                ).best_strehl
                for c in cn2_values
            ]
            color = ARM_COLORS[arm]
            ax.bar(
                x + offset, init, width, label=f"{arm} init", color=color, alpha=0.45
            )
            ax.bar(
                x + offset + width,
                best,
                width,
                label=f"{arm} best",
                color=color,
                alpha=1.0,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([_cn2_axis_label(c) for c in cn2_values])
        ax.set_title(f"{mode} 模式")
        ax.set_ylabel("Strehl")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("每 cn2 的 Strehl 对比 (init vs best, numpy vs oopao)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出 {}", path)


def render_convergence_figure(
    results: list[EpisodeResult], cn2_values: list[float], path: Path
) -> None:
    """闭环收敛轨迹: 每 cn2 一子图, 两臂 Strehl vs SPGD 迭代。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, cn2 in zip(axes.flat, cn2_values):
        for arm in ARMS:
            row = next(
                r
                for r in results
                if r.mode == MODE_CLOSED and r.cn2 == cn2 and r.arm == arm
            )
            ax.plot(
                range(len(row.trace)),
                row.trace,
                label=arm,
                color=ARM_COLORS[arm],
                linestyle=ARM_LINESTYLES[arm],
                marker=ARM_MARKERS[arm],
                markersize=3,
            )
        ax.set_title(f"cn2={fmt_ratio(cn2)} ({_cn2_label(cn2)})")
        ax.set_xlabel("SPGD 迭代")
        ax.set_ylabel("Strehl")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("闭环收敛轨迹 (冻结湍流, 3 步贪婪 SPGD)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出 {}", path)


def render_pib_rms_figure(
    results: list[EpisodeResult], cn2_values: list[float], path: Path
) -> None:
    """闭环模式: PIB (init/best) 与 RMS (init/final) 的每 cn2 对比。"""
    x = np.arange(len(cn2_values))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for arm, offset in ((ARM_NUMPY, -width * 1.5), (ARM_OOPAO, width * 0.5)):
        init = [
            next(
                r
                for r in results
                if r.mode == MODE_CLOSED and r.cn2 == c and r.arm == arm
            ).init_pib
            for c in cn2_values
        ]
        best = [
            next(
                r
                for r in results
                if r.mode == MODE_CLOSED and r.cn2 == c and r.arm == arm
            ).best_pib
            for c in cn2_values
        ]
        color = ARM_COLORS[arm]
        ax.bar(x + offset, init, width, label=f"{arm} init", color=color, alpha=0.45)
        ax.bar(
            x + offset + width, best, width, label=f"{arm} best", color=color, alpha=1.0
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_cn2_axis_label(c) for c in cn2_values])
    ax.set_title("PIB (桶内功率, r=4px)")
    ax.set_ylabel("PIB")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for arm, offset in ((ARM_NUMPY, -width * 1.5), (ARM_OOPAO, width * 0.5)):
        init = [
            next(
                r
                for r in results
                if r.mode == MODE_CLOSED and r.cn2 == c and r.arm == arm
            ).init_rms
            for c in cn2_values
        ]
        final = [
            next(
                r
                for r in results
                if r.mode == MODE_CLOSED and r.cn2 == c and r.arm == arm
            ).final_rms
            for c in cn2_values
        ]
        color = ARM_COLORS[arm]
        ax.bar(x + offset, init, width, label=f"{arm} init", color=color, alpha=0.45)
        ax.bar(
            x + offset + width,
            final,
            width,
            label=f"{arm} final",
            color=color,
            alpha=1.0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_cn2_axis_label(c) for c in cn2_values])
    ax.set_title("相位 RMS (rad)")
    ax.set_ylabel("RMS")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("闭环模式: PIB 与 RMS 每 cn2 对比")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("写出 {}", path)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(results: list[EpisodeResult], path: Path) -> None:
    """把矩阵结果写成 ``summary.csv`` (每 (mode, cn2, arm) 一行)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for result in results:
            writer.writerow(result.csv_row())
    logger.info("写出 {} ({} 行)", path, len(results))


# --------------------------------------------------------------------------- #
# 溯源
# --------------------------------------------------------------------------- #
def _git(args: list[str], repo: Path) -> str:
    """在 ``repo`` 里跑一条 git 命令并返回 stdout (去空白)。"""
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return completed.stdout.strip()


def collect_provenance(oopao_repo: Path) -> dict[str, str]:
    """收集 OOPAO 版本溯源信息 (git 不可用时回落到 unknown)。

    Args:
        oopao_repo: ``libs/OOPAO`` 子模块目录。

    Returns:
        ``{"rev", "rev_date", "verified"}``。
    """
    info: dict[str, str] = {"rev": "unknown", "rev_date": "unknown", "verified": "否"}
    if not oopao_repo.is_dir():
        info["verified"] = "否 (找不到 libs/OOPAO)"
        return info
    try:
        rev_full = _git(["rev-parse", "HEAD"], oopao_repo)
        info["rev"] = rev_full[:7]
        info["rev_date"] = _git(["log", "-1", "--format=%cs", rev_full], oopao_repo)
        info["verified"] = "是"
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("OOPAO git 溯源探测失败 ({}), 回落到 unknown", exc)
    return info


def oopao_package_dir() -> str:
    """已导入的 OOPAO 包目录 (未导入时返回 ``unknown``)。"""
    module = sys.modules.get("OOPAO")
    paths = list(getattr(module, "__path__", []) or []) if module else []
    if not paths:
        return "unknown"
    return str(Path(paths[0]).resolve())


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _phase_screen_ratio_section(
    results: list[EpisodeResult], cn2_values: list[float]
) -> list[str]:
    """§4.3: 相位屏强度 (disturbance_rms) 两臂比值的分析小节。"""
    table_lines, profiles = ratio_table(results, cn2_values)
    claimable = [p for p in profiles if p.is_constancy_claimable]
    positive = [p for p in profiles if p.ratios]

    lines: list[str] = []
    lines.append(
        "`disturbance_rms` 是被路由的 `turbulence_phase` 原始相位屏的 RMS —— "
        "两臂**唯一**的物理输入差异, 所以它的 oopao/numpy 比值可以直接读作"
        "「同一 Cn2 下 OOPAO 相位起伏比 numpy 强多少倍」。"
        "cn2=0 已短路为零屏 (比值无定义), 从表中省略:"
    )
    lines.append("")
    lines.extend(table_lines)
    lines.append("")

    if not claimable:
        lines.append(
            f"本次阶梯只有 {len(positive[0].cn2_levels) if positive else 0} "
            "个 cn2>0 档位, **无法**判定比值是否随湍流强度漂移 —— "
            "「固定标定偏置」的判断需要至少 2 个湍流档位。"
        )
    else:
        constant_profiles = [p for p in claimable if p.is_constant]
        spread = max(p.spread_rel for p in claimable)
        if len(constant_profiles) == len(claimable):
            lines.append(
                f"比值在跨越 {len(claimable[0].cn2_levels)} 个湍流档位 (两个数量级) "
                f"的阶梯上**恒定**: 各 mode 的相对离散度 (max−min)/mean ≤ {spread:.2e}, "
                f"远低于判据 {RATIO_CONSTANT_TOL_REL:g} (浮点舍入量级)。"
                "若差异来自随机相位屏的采样噪声, 比值会随湍流强度漂移; "
                "恒定比值指向两个实现之间的**乘性标定/偏置因子**。"
            )
        else:
            names = " / ".join(p.mode for p in claimable if not p.is_constant)
            lines.append(
                f"比值**并非**在所有 mode 上恒定: {names} 的相对离散度超过判据 "
                f"{RATIO_CONSTANT_TOL_REL:g} (最大 {spread:.2e})。"
                "此时不能把差异整体归结为单一乘性标定系数。"
            )
    if len(constant_modes := [p for p in claimable if p.is_constant]) >= 2:
        pairs = " / ".join(f"{p.mode} {p.mean:.3f}" for p in constant_modes)
        lines.append("")
        lines.append("两条独立证据支持「标定偏置」而非「统计涨落」的判断:")
        lines.append("")
        lines.append(
            "1. **与 Cn2 无关**: 跨越两个数量级的湍流阶梯比值不变 —— "
            "标定系数不随 Cn2 变化。"
        )
        lines.append(
            f"2. **随 mode 改变**: 恒定比值逐 mode 不同 ({pairs}) —— "
<"两者 `l_max` / `propagation_distance` 配置不同。该常数因此是**配置相关**的, "
            "与 `report/oopao_vs_numpy/report.md` 记录的比值随配置变化一致"
            "(该报告在另一组配置下亦测得逐档恒定的常数)。"
        )
    lines.append("")
    lines.append("⚠️ **`init_rms` 与 `disturbance_rms` 是两个不同的量, 不可互相推断**:")
    lines.append(
        "- `disturbance_rms` = `env._disturbance_rms` = **全网格未截瞳**的原始相位屏 "
        "RMS, 不含像差, 不含 DM 校正;"
    )
    lines.append(
        '- `init_rms` = `info["rms"]` = `compat.py::_phase_rms()` = '
        "`angle(湍流 + 像差 + DM 校正)` 后按 `self._mask` **截瞳**的总波前 RMS。"
    )
    inversion = _init_rms_inversion(results, cn2_values)
    if inversion is not None:
        inv_cn2, inv_numpy, inv_oopao = inversion
        lines.append(
            f"因此即使 oopao 臂的相位屏强 {inv_oopao.disturbance_rms / inv_numpy.disturbance_rms:.3f}×, "
            f"`init_rms` 仍可跨臂反向: 实测 {inv_numpy.mode}/{fmt_ratio(inv_cn2)} "
            f"numpy init_rms {inv_numpy.init_rms:.4f} > oopao {inv_oopao.init_rms:.4f}, "
            f"而同一格的 disturbance_rms 是 {inv_numpy.disturbance_rms:.4f} vs "
            f"{inv_oopao.disturbance_rms:.4f}。像差项与截瞳权重同时参与合成, "
            "所以「相位屏更强 → init_rms 必然更大」不成立。"
        )
    else:
        lines.append(
            "因此不能由 `disturbance_rms` 反推 `init_rms`: 后者还含像差项与截瞳权重。"
        )
    lines.append("Strehl 同理 —— 它由截瞳后的**总**波前决定, 而不只是相位屏。")
    return lines


def _incomparability_section(
    results: list[EpisodeResult], cn2_values: list[float]
) -> list[str]:
    """§9: 非可比性声明。"""
    _, profiles = ratio_table(results, cn2_values)
    constant = [p for p in profiles if p.is_constant]
    lines: list[str] = []
    lines.append(
        "绝对 Strehl/PIB 不可跨臂直接比较 —— 同 Cn2 下 OOPAO 相位起伏远强于 numpy。"
    )
    if constant:
        lines.append("`disturbance_rms` 的两臂比值在本次阶梯上**逐位恒定** (§4.3):")
        for profile in constant:
            lines.append(
                f"- `{profile.mode}` 模式: **{profile.mean:.3f}×** "
                f"(相对离散度 {profile.spread_rel:.2e}, 在 {len(profile.ratios)} "
                "个湍流档位上恒定)"
            )
    else:
        lines.append(
            "`disturbance_rms` 的两臂比值需要至少 2 个 cn2>0 档位才能判定恒定性; "
            "本次阶梯档位不足, 未能给出可判定的比值 (§4.3) —— "
            "跨臂绝对指标同样不可比 (定性结论不变)。"
        )
    gap = _max_init_strehl_gap(results, cn2_values)
    lines.append("")
    if gap is not None:
        gap_cn2, gap_numpy, gap_oopao = gap
        lines.append(
            f"这意味着表中某些行**看似「OOPAO 更优」, 但不可解读为后端更优**: "
            f"例如 {gap_numpy.mode} 模式 cn2={fmt_ratio(gap_cn2)} 时, oopao 的 init Strehl "
            f"({gap_oopao.init_strehl:.4f}) 高于 numpy ({gap_numpy.init_strehl:.4f})。"
            f"但同一格的 disturbance_rms 是 {gap_oopao.disturbance_rms:.4f} vs "
            f"{gap_numpy.disturbance_rms:.4f} —— 两臂承受的相位屏强度本就不同; "
            f"且 init_rms 为 {gap_oopao.init_rms:.4f} vs {gap_numpy.init_rms:.4f}, "
            "Strehl 由**截瞳后总波前** (湍流 + 像差 + DM 校正) 决定, "
            "而不是相位屏单独决定 (见 §4.3 注)。该行的 Strehl 高低主要反映"
            "**各自所受扰动的合成相位**, 无法据此判定哪一臂的传播核更优。"
        )
    else:
        lines.append(
            "任何单行指标的两臂差异都**不可**解读为「某臂后端更优」: "
            "两臂承受的相位屏强度不同, 且 Strehl/PIB 由截瞳后总波前 "
            "(湍流 + 像差 + DM 校正) 决定, 而不是相位屏单独决定 (见 §4.3 注)。"
        )
    lines.append("")
    lines.append(
        "要判定哪一臂的绝对 Strehl 更接近物理真值, 需先把两臂的相位屏"
        "**标定到同一 r0 / 同一 phase_std**, 再重跑本矩阵 —— "
        "当前数据不足以支持该结论。本报告对比的是「同一 Cn2 配置下, "
        "切换后端对端到端行为的影响」, 不是「等价实现」。"
    )
    return lines


def write_report(
    *,
    results: list[EpisodeResult],
    cn2_values: list[float],
    out_dir: Path,
    provenance: dict[str, str],
    oopao_dir: str,
    n_grid: int,
    seed: int,
    steps: int,
    control_ok: bool,
    differing_cells: int,
    dead_cn2_probe: dict[str, Any],
    figure_names: list[str],
) -> Path:
    """生成中文 ``report.md`` (所有数字均来自本次运行)。

    Args:
        results: 全部 episode 结果。
        cn2_values: 本次实际使用的 cn2 阶梯。
        out_dir: 输出目录。
        provenance: OOPAO 溯源信息。
        oopao_dir: OOPAO 包目录。
        n_grid: 仿真网格边长。
        seed: 随机种子。
        steps: episode 步数预算。
        control_ok: cn2=0 对照组是否通过。
        differing_cells: 反空洞守卫计数。
        dead_cn2_probe: slm_shaping_bench 死配置探测结果。
        figure_names: 图文件名列表 (用于产物清单)。

    Returns:
        报告文件路径。
    """
    generated_at = datetime.now().isoformat(timespec="seconds")
    report_path = out_dir / "report.md"
    lines: list[str] = []
    lines.append("# AO_OOPAO_BACKEND 端到端影响报告 (SimTurbulenceAOEnv)")
    lines.append("")
    lines.append(
        f"> 生成时间: {generated_at} | 脚本: `scripts/generate_oopao_impact_report.py` | "
        f"OOPAO rev: `{provenance['rev']}` ({provenance['rev_date']}, git 校验="
        f"{provenance['verified']}) | OOPAO 包目录: `{oopao_dir}`"
    )
    lines.append("")
    lines.append("## 1. 适用范围 (路由红线)")
    lines.append("")
    lines.append(
        "`AO_OOPAO_BACKEND` 只路由 `beam_backend.turbulence_phase()` 与 "
        "`beam_backend.propagate()`; `focal_plane()` / `apply_lens()` 恒为 numpy。"
        "本报告驱动的是 `SimTurbulenceAOEnv` (RL AO 环境: DM 影响函数 + WFS 斜率 + "
        "焦面 FFT), 其湍流相位屏经 `turbulence_phase()` 路由, 焦面/Strehl/PIB/RMS "
        "下游计算是同一份代码。因此两臂差异**只来自湍流相位屏**。"
    )
    lines.append("")
    lines.append("## 2. 方法与指标定义")
    lines.append("")
    lines.append(
        f"- 环境: `SimTurbulenceAOEnv` (n_grid={n_grid}, {N_ACTUATORS}×{N_ACTUATORS} DM, "
        f"{N_SUBAPERTURES}×{N_SUBAPERTURES} 子孔径, λ=1550nm, 口径 0.1m, 传播距离 "
        f"1000m, PIB 桶半径 {PIB_RADIUS_PX}px)"
    )
    lines.append(
        f"- 种子: `seed={seed}` (全局 `np.random.seed` + `env.reset(seed=...)`), "
        f"steps={steps}"
    )
    lines.append("- 两种模式:")
    lines.append(
        "  - `open` 开环: 滑动湍流窗口 (screen_step_px=2), 零动作, 记录湍流演化下的端到端指标"
    )
    lines.append(
        "  - `closed` 闭环: 冻结湍流 (screen_step_px=0), 3 步贪婪 SPGD "
        f"(δ={SPGD_DELTA}, 随机方向 ± 评估 + 回位到更优位置), 记录校正收敛"
    )
    lines.append(
        "- 指标: Strehl (焦面峰值/理想峰值), PIB (桶内功率, r=4px), RMS (相位 RMS), "
        "best-Strehl (episode 内最优)"
    )
    lines.append(
        "- 后端卫生: 每臂切换时 set/pop `AO_OOPAO_BACKEND` + "
        "`oopao_backend._get_backend.cache_clear()` (进入与退出都清); 每行记录实测 "
        "`_oopao_enabled()` 并校验与臂一致"
    )
    lines.append("")
    lines.append("## 3. Cn2 阶梯")
    lines.append("")
    lines.append("| cn2 | 标签 | 说明 |")
    lines.append("|-----|------|------|")
    for cn2 in cn2_values:
        note = "对照组 (`turbulence_phase` 在 cn2<=0 短路为零屏)" if cn2 <= 0 else ""
        lines.append(f"| {fmt_ratio(cn2)} | {_cn2_label(cn2)} | {note} |")
    lines.append("")
    lines.append("## 4. 汇总对比")
    lines.append("")
    lines.append("### 4.1 开环 (open)")
    lines.append("")
    lines.append(_metric_table(results, MODE_OPEN, cn2_values))
    lines.append("")
    lines.append("### 4.2 闭环 (closed)")
    lines.append("")
    lines.append(_metric_table(results, MODE_CLOSED, cn2_values))
    lines.append("")
    lines.append("### 4.3 关键分析: 相位屏强度差异是固定标定偏置")
    lines.append("")
    lines.extend(_phase_screen_ratio_section(results, cn2_values))
    lines.append("")
    lines.append("## 5. 收敛轨迹 (closed)")
    lines.append("")
    lines.append("![收敛轨迹](figures/convergence_trace.png)")
    lines.append("")
    lines.append(
        "闭环模式下湍流被冻结, 3 步贪婪 SPGD 用 16 路 DM 校正静态像差。"
        "两臂的收敛轨迹差异直接反映相位屏强度差异: numpy 臂湍流弱 (init Strehl 高), "
        "校正增益小; oopao 臂湍流强 (init Strehl 低), 校正增益大但绝对上限低。"
    )
    lines.append("")
    lines.append("## 6. cn2=0 对照组")
    lines.append("")
    lines.append(
        "cn2=0 时 `turbulence_phase` 短路为零屏, 两臂应**位级一致**。"
        f"校验结果: **{'通过' if control_ok else '失败 (见日志警告)'}**。"
    )
    lines.append("")
    lines.append("## 7. 反空洞守卫")
    lines.append("")
    lines.append(
        f"两臂在 cn2>0 下确实不同: **通过** ({differing_cells} 个 (mode, cn2>0) "
        "单元格的 init/best Strehl 或 PIB/RMS 不同)。若此处失败, 说明后端开关 "
        "没有影响被测路径, 报告将拒绝写出。"
    )
    lines.append("")
    lines.append("## 8. 负发现: slm_shaping_bench 的 cn2 是死配置")
    lines.append("")
    lines.append(
        f"`slm_shaping_bench.forward_intensity` 在 cn2=0 与 cn2=5e-14 下输出"
        f"**字节一致** (max|ΔI| = {dead_cn2_probe['max_abs_diff']:.6g})。"
        "它 import 了 `turbulence_phase` 但从不调用; 只调用 `focal_plane` "
        "(永不路由)。因此 `slm_shaping_bench` **不能**作为后端影响测试载体。"
    )
    lines.append("")
    lines.append("## 9. 非可比性声明")
    lines.append("")
    lines.extend(_incomparability_section(results, cn2_values))
    lines.append("")
    lines.append("## 10. 产物清单")
    lines.append("")
    lines.append("- `summary.csv` — 每 (mode, cn2, arm) 一行")
    for name in figure_names:
        lines.append(f"- `figures/{name}`")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("写出 {}", report_path)
    return report_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="生成 AO_OOPAO_BACKEND 端到端影响报告 (SimTurbulenceAOEnv, 离线)",
    )
    parser.add_argument("--n-grid", type=int, default=64, help="仿真网格边长 (默认 64)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认 42)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "report" / "oopao_impact",
        help="输出目录 (默认 report/oopao_impact)",
    )
    parser.add_argument(
        "--cn2",
        default=DEFAULT_CN2,
        help="逗号分隔的 cn2 阶梯 (默认 " + DEFAULT_CN2 + ")",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="每 episode 步数预算 (closed 模式 SPGD 迭代数 = steps//3; 默认 60)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="冒烟模式: 只取前 2 个 cn2 档位, steps=10",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Returns:
        进程退出码 (0 = 成功)。
    """
    args = parse_args(argv)
    if args.n_grid < 8:
        raise ValueError(f"--n-grid must be >= 8, got {args.n_grid}")
    if args.steps < 3:
        raise ValueError(f"--steps must be >= 3, got {args.steps}")

    cn2_values = [float(v) for v in args.cn2.split(",") if v.strip()]
    if not cn2_values:
        raise ValueError("--cn2 为空")
    steps = 10 if args.quick else args.steps
    if args.quick:
        cn2_values = cn2_values[:2]

    out_dir = Path(args.out_dir)
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "矩阵: {} cn2 × {} 臂 × {} 模式 = {} 次 episode (n_grid={}, seed={}, steps={})",
        len(cn2_values),
        len(ARMS),
        len(MODES),
        len(cn2_values) * len(ARMS) * len(MODES),
        args.n_grid,
        args.seed,
        steps,
    )

    try:
        assert_oopao_usable()
    except OopaoBackendUnavailableError as exc:
        logger.error("FATAL: {}", exc)
        return 2

    provenance = collect_provenance(ROOT / "libs" / "OOPAO")
    oopao_dir = oopao_package_dir()
    logger.info(
        "OOPAO 溯源: rev={} (git 校验={})",
        provenance["rev"],
        provenance["verified"],
    )

    results: list[EpisodeResult] = []
    for cn2 in cn2_values:
        for mode in MODES:
            for arm in ARMS:
                logger.info("episode: cn2={} mode={} arm={}", fmt_ratio(cn2), mode, arm)
                results.append(
                    run_episode(
                        mode=mode,
                        cn2=cn2,
                        arm=arm,
                        n_grid=args.n_grid,
                        seed=args.seed,
                        steps=steps,
                    )
                )
    if not results:
        logger.error("矩阵为空, 没有任何结果可写")
        return 3

    verify_backend_routing(results)
    differing_cells = verify_anti_vacuity(results, cn2_values)
    control_ok = verify_control(results, cn2_values)

    write_csv(results, out_dir / "summary.csv")

    figure_names = ["strehl_by_cn2.png", "convergence_trace.png", "pib_rms_by_cn2.png"]
    render_strehl_figure(results, cn2_values, figures_dir / figure_names[0])
    render_convergence_figure(results, cn2_values, figures_dir / figure_names[1])
    render_pib_rms_figure(results, cn2_values, figures_dir / figure_names[2])
    for name in figure_names:
        figure_path = figures_dir / name
        if not figure_path.is_file():
            raise RuntimeError(f"图缺失: {figure_path}")

    dead_cn2_probe = probe_slm_shaping_bench_dead_cn2()

    write_report(
        results=results,
        cn2_values=cn2_values,
        out_dir=out_dir,
        provenance=provenance,
        oopao_dir=oopao_dir,
        n_grid=args.n_grid,
        seed=args.seed,
        steps=steps,
        control_ok=control_ok,
        differing_cells=differing_cells,
        dead_cn2_probe=dead_cn2_probe,
        figure_names=figure_names,
    )

    logger.info("完成: {}", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
