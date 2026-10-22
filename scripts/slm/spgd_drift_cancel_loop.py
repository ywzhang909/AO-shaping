"""SPGD 闭环的照明漂移对消 A/B：数字孪生上的协议级验证 (#144 第 1 项 / #36 验收判据).

Issue 背景
---------
#144 (论文核心阻塞): 实机 SPGD 在三个步长量级全部失败, 根因是台架无法分辨梯度 ——
同一相位重复测量亮度漂 37-45%, 记录分数与 ``max_brt`` 的相关 R²=0.390, 即分数变化
几乎全部来自照明漂移而非目标函数。其最高优先改进是「照明漂移对消」: 对*被优化量本身*
建同相位重复噪声底, 同一连接内交错重放平场/候选相位, 用 flat 帧的漂移趋势归一候选帧,
再算梯度。#36 (H-19) 则给出了已落地 ABBA 采样的实机验收判据 (``dec > 0.55`` 且
``late_gain >= 10%``)。

本脚本做什么
------------
在 2f-Fourier 数字孪生上跑**真实的** ``slm-pib spgd`` 闭环 (Click 入口、真实优化器、
真实 recorder), 对三种测量协议做 A/B:

* ``baseline``  — 优化器原生 ``+ -`` 两帧协议 (漂移照常泄漏进梯度);
* ``abba``      — 已落地的 ``--abba-sampling`` 回文采样 (只对消线性漂移, #36);
* ``flat_norm`` — 本 issue 的第 1.2 项协议: 每轮 ``flat/+d/flat/-d/flat`` 交错重放,
                   候选帧按 flat 帧测得的增益曲线归一 (对消任意形状漂移)。

照明环境模型 (harness 注入, 不改孪生源码)
-----------------------------------------
实测结论 (见 report): 纯倍增增益对孪生的目标函数**不可见** —— shape 目标各项
(EE/CV/峰值比) 都是同缩放比率。要在孪生上复现 bench 的「亮度漂移主导分数」, 需要一个
不随激光功率缩放的通道, 即**杂散光/背景底板** (bench 上同样存在: 总能量分母里的背景、
ROI 里的固定底噪)。因此本 harness 把台架照明建模为::

    帧 = g(t) * 远场信号 + 散粒/读出噪声 + 杂散底板 B        (g: 激光功率漂移)

* ``g(t) = 1 + a_smooth*sin(2*pi*n/T) + a_walk*walk(n)``  —— 平滑分量 (热漂移) +
  反射随机游走 (随机功率波动), ``n`` 为相机读出序号; ABBA 只能对消前者, flat 归一
  两者都能对消 —— 这正是两种协议的机制分野。
* 增益施加在**光学信号**上 (包住 ``SimPibSystem.far_field``), 噪声模型原样在后:
  散粒噪声随信号缩放 (物理正确), 读出噪声保持绝对 (物理正确)。
* 杂散底板 B 加在噪声之后 (传感器上的固定背景)。

判据 (全部来自 issue 原文)
--------------------------
* 同相位噪声底: 同一相位 >= 8 次重复, 建 σ(J); 候选相位相对平场的分数变化须
  > 3σ 才计「可分辨」 (#144 测试方法)。
* ``dec`` (frac_decreasing) > 0.55 且 ``late_gain`` >= 10% (#36 验收判据, 取
  ``tools/kernels/sweep_analysis.analyse_trace`` 的权威实现)。
* R²(J ~ max_brt): baseline 应高 (漂移主导), 归一后应低。

边界
----
* 只读孪生、只 monkeypatch 进程内对象, 不改任何仓库源码; 运行产物写在仓库外。
* 孪生 ``SimPibCCD.get_numpy_image`` 忽略 ``n_sample`` (不平均), 故 ``--n-eval-frames``
  在孪生上是空操作 —— 真实相机才会平均。本报告据此按单帧读出解释所有 σ。

用法::

    python scripts/slm/spgd_drift_cancel_loop.py --smoke      # 2 epoch 接线自检
    python scripts/slm/spgd_drift_cancel_loop.py --epochs 40  # 完整 A/B 矩阵
    # 本底 σ 探针 (drift off, 杂散底板开): 给 3σ 判据一个基准行
    python scripts/slm/spgd_drift_cancel_loop.py --variants baseline --epochs 12 \\
        --a-smooth 0 --walk-sigma 0 --data-root <scratch>/intrinsic
    # 修 compute_drift_verdict_metrics 后不重跑矩阵: 从已有 pkl 重算指标并重写双落点报告
    python scripts/slm/spgd_drift_cancel_loop.py --recompute \\
        --data-root <scratch>/full --intrinsic <scratch>/intrinsic/results.json

报告双落点 (report/AGENTS.md §3: 路径常量写死): 仓库外 ``<data-root>/report.md``
速览 + 仓库内 ``report/shape_trials/spgd_drift_cancel_loop/`` (带日期戳的图与
``artifacts/results.json`` 输入契约)。
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "libs"))

import numpy as np
from loguru import logger

from ao_shaping.drivers.sim.slm_pib_sim import (
    SimPibSystem,
    register_sim_camera,
    reset_system,
)
from ao_shaping.drivers.sim.sim_bench_patch import install_sim_slm
from ao_shaping.tools.kernels.sweep_analysis import analyse_trace

#: 孪生光学系统种子 (与 ``slm_pib_sim_run.py`` 的 SYSTEM_SEED 一致)。
SYSTEM_SEED = 42

#: 漂移模型种子: 四个变体共用同一条 g(n) 序列 (按读出序号), 保证可比。
DRIFT_SEED = 20261022

#: 默认漂移参数: 平滑热漂移 12% / 周期 61 次读出; 随机游走 σ=2%/次, 截断 ±30%。
#: 按 32 epoch x ~6 次读出/epoch ≈ 190 次读出缩放: 游走标准差 ~0.28, 加正弦后
#: 全程亮度摆幅 ~50%, 与 bench「同一相位重复测量亮度漂 37-45%」同量级。
DEFAULT_A_SMOOTH = 0.12
DEFAULT_T_SMOOTH = 61.0
DEFAULT_WALK_SIGMA = 0.02
DEFAULT_WALK_CLIP = 0.30

#: 运行产物根目录的默认暂存位置: 系统临时目录 (仓库外, 机器无关)。写死某台开发机的
#: 绝对路径在非 Windows 上会退化成相对目录 `./D:/...` —— 脚本照常跑完, 产物却落在
#: 没人查看的地方 (守卫: tests/ao_shaping/scripts/test_script_repo_root_bootstrap.py)。
#: 每次实验用 --data-root 指到该次实验自己的暂存目录。
DEFAULT_DATA_ROOT = Path(tempfile.gettempdir()) / "ao_spgd_drift_scratch"

#: 默认杂散底板 [ADU/px]。实测 (report): 该值下 R²(J~max_brt)≈0.59、同相位亮度
#: 摆幅 ~34%, 与 bench 的 37-45% / R²=0.39 同量级; B=0 时孪生目标函数对纯倍增
#: 增益完全免疫 (各项同缩放), 复现不出 #144 的失效模式。
DEFAULT_B_PX = 0.005

#: 平场参考相位重复次数下限 (建同相位噪声底, #144 要求 >= 5, 这里取 8)。
MIN_REF_REPEATS = 8

#: 增益估计量: 参考帧最亮的 N 个像素均值 (抗单像素噪声; 峰值法实测 ~1.6% 抖动)。
GAIN_TOP_K = 64

#: 报告落点 (report/AGENTS.md §2.2: 新脚本 → 新子目录, 目录名 = 脚本语义名;
#: 路径常量写死, 不用「写到别处再手工搬」)。
REPORT_KEY = "report/shape_trials/spgd_drift_cancel_loop/spgd_drift_cancel_loop_20261022.md"
OUT_DIR = ROOT / "report" / "shape_trials" / "spgd_drift_cancel_loop"
FIG_DIR = OUT_DIR / "figures"
ARTIFACT_DIR = OUT_DIR / "artifacts"

#: 图名日期戳, 与报告名后缀一致 (report/AGENTS.md §2.1: 同系列每份报告独占图)。
FIG_STAMP = "20261022"

#: 优化器噪声门参数 (``SlmZernikePibConfig`` 默认值, 本 harness 未覆盖):
#: ``|diff| <= k * sigma(前序 window 个 diff)`` 即零置本轮更新。
NOISE_GATE_K = 3.0
NOISE_GATE_WINDOW = 20


# === SECTION: 照明环境模型 =================================================


class IlluminationDrift:
    """激光功率漂移 + 杂散底板的台架照明环境模型 (进程内注入孪生)。

    增益按**相机读出序号** ``n`` 演化 (与协议读取速率解耦, 四个变体看到同一条
    g(n) 序列)::

        g(n) = 1 + a_smooth*sin(2*pi*n/T) + a_walk*walk(n)
        walk(n) = clip(walk(n-1) + sigma*N(0,1), +-clip)

    平滑分量模拟热漂移 (ABBA 可对消的线性局部近似), 随机游走模拟功率波动
    (只有 flat 归一能对消)。轨迹全程记录, 供报告复核摆动幅度与估计量保真度。
    """

    def __init__(
        self,
        *,
        a_smooth: float = DEFAULT_A_SMOOTH,
        t_smooth: float = DEFAULT_T_SMOOTH,
        walk_sigma: float = DEFAULT_WALK_SIGMA,
        walk_clip: float = DEFAULT_WALK_CLIP,
        stray_adu: float = DEFAULT_B_PX,
        seed: int = DRIFT_SEED,
    ) -> None:
        """构造照明环境模型。

        Args:
            a_smooth: 平滑 (正弦) 漂移幅度, 相对值。
            t_smooth: 平滑漂移周期, 单位读出次数。
            walk_sigma: 随机游走每步 σ, 相对值。
            walk_clip: 随机游走截断 (相对值), 防止增益漂离物理合理区间。
            stray_adu: 杂散底板 [ADU/px], 传感器固定背景。
            seed: 漂移 RNG 种子 (四个变体共用)。
        """
        self.a_smooth = float(a_smooth)
        self.t_smooth = float(t_smooth)
        self.walk_sigma = float(walk_sigma)
        self.walk_clip = float(walk_clip)
        self.stray_adu = float(stray_adu)
        self._rng = np.random.default_rng(seed)
        self._walk = 0.0
        self._n = 0
        self.trace: list[dict[str, float]] = []

    @property
    def n_reads(self) -> int:
        """已发生的相机读出次数 (增益时钟)。"""
        return self._n

    def gain(self) -> float:
        """当前增益 (每次相机读出自增一次时钟)。"""
        self._n += 1
        self._walk += float(self._rng.normal(0.0, self.walk_sigma))
        self._walk = float(np.clip(self._walk, -self.walk_clip, self.walk_clip))
        g = 1.0 + self.a_smooth * np.sin(
            2.0 * np.pi * self._n / self.t_smooth
        ) + self._walk
        self.trace.append({"n": float(self._n), "g": float(g)})
        return float(g)

    def install(self, system: SimPibSystem) -> None:
        """把漂移 + 杂散底板注入进程级孪生系统 (只包进程内对象)。

        增益包在 ``far_field`` (光学信号) 上, 杂散底板包在 ``far_field_noisy``
        (传感器) 上 —— 于是散粒噪声随信号缩放、读出噪声与底板保持绝对, 与真实
        台架的光路/传感器分工一致。
        """
        if getattr(system.far_field, "_drift_wrapper", False):
            return
        original_far_field = system.far_field
        original_noisy = system.far_field_noisy

        def drifted_far_field() -> np.ndarray:
            return original_far_field() * self.gain()

        def drifted_noisy() -> np.ndarray:
            return original_noisy() + self.stray_adu

        drifted_far_field._drift_wrapper = True  # type: ignore[attr-defined]
        system.far_field = drifted_far_field  # type: ignore[method-assign]
        system.far_field_noisy = drifted_noisy  # type: ignore[method-assign]

    def swing_pct(self) -> float:
        """已记录增益轨迹的峰峰摆幅 [%] (对照 bench 的 37-45%)。"""
        if not self.trace:
            return 0.0
        gs = np.array([t["g"] for t in self.trace], dtype=np.float64)
        return float(100.0 * (gs.max() - gs.min()) / gs.mean())


# === SECTION: 平场归一化测量协议 ===========================================

#: 当前协议模式: "off" (不 patch) / "observe" (只记录 flat 帧) / "normalize"。
_PROTOCOL_MODE = "off"
_DRIFT: IlluminationDrift | None = None
_WRAPPER_STATE: dict[str, Any] = {}


def _topk_mean(img: np.ndarray, k: int = GAIN_TOP_K) -> float:
    """最亮 k 个像素的均值 —— 抗单像素噪声的增益估计量。"""
    flat = np.sort(np.asarray(img, dtype=np.float64).ravel())
    return float(flat[-k:].mean())


class _FlatNormWrapper:
    """``BenchSession.capture`` 的协议级包装 (平场交错重放 + 归一)。

    ``observe`` 模式: 每个候选帧后补一次平场参考帧 (命中远场缓存, 只重采噪声),
    用**优化器自己的目标函数**给参考帧打分并记录 —— 于是同相位噪声底 σ(J) 与
    闭环用的是同一个 scorer, 无镜像偏差; 候选帧原样返回, 对优化器行为零影响。

    ``normalize`` 模式 (即 #144 第 1.2 项的「简单改动」): 每个候选帧用前后两个
    参考帧的增益线性插值归一::

        帧 = g(t)*信号 + 噪声 + B;  ref 帧测得亮度 m(t) = g(t)*S_ref + B
        img_norm = img * (SCALE / m(t_cand))

    候选帧的信道部分被拉回同一照明标度, 漂移不再泄漏进 pos−neg 差分; B 导致的
    二阶残余保留 (与 bench 上同款残余, 对应 #144 第 2 项的背景扣除)。
    """

    def __init__(self, session: Any, original: Any, drift: IlluminationDrift) -> None:
        self._session = session
        self._original = original
        self._drift = drift
        self._ref_coeffs: np.ndarray | None = None
        self._scale: float | None = None
        self._last_ref_gain: float | None = None
        self._last_ref_n: int | None = None
        self.captures: list[dict[str, Any]] = []

    # -- 内部: 一次原始采集 (显示相位 -> 等待 -> 读帧) --------------------

    def _raw_capture(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._original(coeffs)

    def _score(self, img: np.ndarray) -> float:
        """用优化器自己的目标函数给一帧打分 (无守卫副作用, 不写任何状态)。"""
        j, _ratio, _tracking = self._session._shaping.raw(img)
        return float(j)

    def _ref_capture(self) -> tuple[float, float, float, int]:
        """采集一帧平场参考; 返回 (测得亮度, J, 真实增益, 读出序号)。"""
        assert self._ref_coeffs is not None
        _c, _g, img = self._raw_capture(self._ref_coeffs)
        measured = _topk_mean(img)
        return measured, self._score(img), self._drift.trace[-1]["g"], self._drift.n_reads

    def __call__(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """协议入口: 签名与 ``BenchSession.capture`` 一致。"""
        if _PROTOCOL_MODE == "off":
            return self._raw_capture(coeffs)

        if self._ref_coeffs is None:
            # 首个候选 (epoch 1 的 +d): 顺手把它固定为永久参考相位。参考相位只需
            # 远场稳定, 不必是平场零点 —— 它测的是照明, 不是形状。
            clipped, gray, img = self._raw_capture(coeffs)
            self._ref_coeffs = np.array(clipped, dtype=np.float64, copy=True)
            measured, j_ref, g_true, n = self._ref_capture()
            self._scale = measured
            self._last_ref_gain = measured
            self._last_ref_n = n
            self.captures.append(
                {
                    "kind": "establish",
                    "n": n,
                    "g_true": g_true,
                    "measured": measured,
                    "j_ref": j_ref,
                    "normalized": False,
                }
            )
            return clipped, gray, img

        if _PROTOCOL_MODE == "observe":
            clipped, gray, img = self._raw_capture(coeffs)
            measured, j_ref, g_true, n = self._ref_capture()
            self._last_ref_gain = measured
            self._last_ref_n = n
            self.captures.append(
                {
                    "kind": "observe",
                    "n": n,
                    "g_true": g_true,
                    "measured": measured,
                    "j_ref": j_ref,
                    "normalized": False,
                }
            )
            return clipped, gray, img

        # normalize: 候选帧夹在两个参考帧之间, 按插值增益归一。
        pre_measured, pre_j, _pre_g, pre_n = self._ref_capture()
        clipped, gray, img = self._raw_capture(coeffs)
        # 候选读出的增益/序号必须紧邻候选采集记录 (之后的参考帧会推进时钟)。
        g_cand = self._drift.trace[-1]["g"]
        n_cand = self._drift.n_reads
        post_measured, post_j, _post_g, post_n = self._ref_capture()

        span = float(post_n - pre_n)
        frac = 0.5 if span <= 0 else float(n_cand - pre_n) / span
        gain_at_cand = pre_measured + frac * (post_measured - pre_measured)
        assert self._scale is not None and self._scale > 0.0
        norm_img = img * (self._scale / gain_at_cand) if gain_at_cand > 0 else img

        self.captures.append(
            {
                "kind": "cand",
                "n": n_cand,
                "g_true": g_cand,
                "measured_pre": pre_measured,
                "measured_post": post_measured,
                "gain_used": gain_at_cand,
                "j_ref_pre": pre_j,
                "j_ref_post": post_j,
                "normalized": True,
            }
        )
        return clipped, gray, norm_img

    def same_phase_sigma(self) -> dict[str, float]:
        """同相位重复的 J 统计 (参考帧即同相位重复测量, 与闭环同一 scorer)。"""
        js: list[float] = []
        for c in self.captures:
            for key in ("j_ref", "j_ref_pre", "j_ref_post"):
                value = c.get(key)
                if value is not None:
                    js.append(float(value))
        arr = np.asarray(js, dtype=np.float64)
        if arr.size < 2:
            return {"n": float(arr.size), "sigma": float("nan"), "mean": float("nan")}
        return {
            "n": float(arr.size),
            "sigma": float(arr.std(ddof=1)),
            "mean": float(arr.mean()),
        }


def install_capture_protocol(mode: str, drift: IlluminationDrift) -> None:
    """把协议包装装到 ``BenchSession.capture`` (类级, 每会话重建状态)。"""
    global _PROTOCOL_MODE, _DRIFT, _WRAPPER_STATE
    _PROTOCOL_MODE = mode
    _DRIFT = drift
    _WRAPPER_STATE = {}

    import ao_shaping.optimizer.wfless.slm_zernike_pib as opt

    if getattr(opt.BenchSession.capture, "_protocol_wrapper", False):
        return
    original_capture = opt.BenchSession.capture

    def patched(self: Any, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 注意: drift 必须从模块级 ``_DRIFT`` 读取 (每个变体 install 一个新模型),
        # 不能沿用首次 install 的闭包 —— 否则后续变体会用第一个变体的漂移序列。
        state = _WRAPPER_STATE.get("wrapper")
        if state is None or state["session"] is not self:
            state = {
                "session": self,
                "wrapper": _FlatNormWrapper(self, original_capture.__get__(self), _DRIFT),
            }
            _WRAPPER_STATE["wrapper"] = state
        return state["wrapper"](coeffs)

    patched._protocol_wrapper = True  # type: ignore[attr-defined]
    opt.BenchSession.capture = patched  # type: ignore[method-assign]


def current_wrapper() -> _FlatNormWrapper | None:
    """当前会话的协议包装 (若已安装)。"""
    state = _WRAPPER_STATE.get("wrapper")
    return None if state is None else state["wrapper"]


# === SECTION: 闭环变体运行 =================================================


def _click_args(data_root: Path, epochs: int, *, abba: bool) -> list[str]:
    """构造一次 ``slm-pib spgd`` 孪生运行的 CLI 参数。

    lr/delta 全部显式钉住 (``--lr 0.3`` 非零, 调度器不接管), 四个变体共享同一条
    更新规则 —— 唯一变量是测量协议。曝光固定 80 ms (自动曝光会补偿漂移, 属另一种
    对消机制, 不在本实验内); 目标 100x100、3 帧 (孪生上为空操作, 见模块 docstring)。
    """
    args = [
        "spgd",
        "-d",
        str(data_root),
        "--debug",
        "--cam-type",
        "sim",
        "--cam-id",
        "0",
        "--exposure-time-ms",
        "80",
        "--cam-size",
        "512",
        "-c",
        "shape",
        "--slm-number",
        "1",
        "--slm-wavelength",
        "1064",
        "-n",
        "4",
        "--objective",
        "shape",
        "--target-shape",
        "square",
        "--target-size",
        "100",
        "--target-aspect-ratio",
        "1.0",
        "-e",
        str(epochs),
        "--delta",
        "0.2",
        "--lr",
        "0.3",
        "--optimizer-type",
        "adamod",
        "--w-uniformity",
        "2.0",
        "--w-peak",
        "0.5",
    ]
    if abba:
        args.append("--abba-sampling")
    return args


def _artifact_roots(data_root: Path) -> set[Path]:
    """``<data_root>/debug`` 下现有的 ``slm_pib_*`` **时间戳运行目录**集合。

    粒度必须取到时间戳子目录 (``slm_pib_<obj>/<stamp>/``): 家族目录在首次运行后
    就已存在, 取家族目录会让第二个及以后的变体永远落在 before/after 差集之外。
    """
    debug_root = Path(data_root) / "debug"
    if not debug_root.is_dir():
        return set()
    return {
        p.resolve()
        for family in debug_root.glob("slm_pib_*")
        if family.is_dir()
        for p in family.glob("*")
        if p.is_dir()
    }


def _newest_pkl(data_root: Path, before: set[Path]) -> Path | None:
    """本次运行新产生的 recorder pkl (before/after 差集, 抗并发)。

    产物布局是 ``<root>/debug/slm_pib_<obj>/<stamp>/<stem>.pkl``, 时间戳目录即
    一次运行的根, pkl 直接在其下。
    """
    after = _artifact_roots(data_root)
    new = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    root = new[0] if new else None
    if root is None:
        return None
    pkls = sorted(root.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pkls[0] if pkls else None


def _load_rows(pkl_path: Path) -> list[dict[str, Any]]:
    """读取 recorder pkl (``{epoch: row}`` 字典) 为行列表。"""
    payload = pickle.loads(pkl_path.read_bytes())
    if isinstance(payload, dict):
        return [dict(v) for v in payload.values() if isinstance(v, dict)]
    if isinstance(payload, list):
        return [dict(r) for r in payload]
    raise TypeError(f"unexpected recorder payload type: {type(payload).__name__}")


def run_variant(
    name: str,
    *,
    epochs: int,
    drift: IlluminationDrift | None,
    protocol: str,
    abba: bool,
    data_root: Path,
) -> dict[str, Any]:
    """跑一个闭环变体, 返回结果字典 (含指标)。

    Args:
        name: 变体名 (``control`` / ``baseline`` / ``abba`` / ``flat_norm``)。
        epochs: SPGD 轮数。
        drift: 照明环境模型; ``None`` 表示无漂移 (control)。
        protocol: 捕获协议模式 (``off`` / ``observe`` / ``normalize``)。
        abba: 是否传 ``--abba-sampling`` (优化器原生 ABBA)。
        data_root: 运行产物根目录 (仓库外)。
    """
    import ao_shaping.optimizer.wfless.slm_zernike_pib as opt
    from ao_shaping.runners.slm.slm_shaping_runner import run as slm_pib_run

    system = reset_system(seed=SYSTEM_SEED)
    assert isinstance(system, SimPibSystem)
    if drift is not None:
        drift.install(system)
    install_sim_slm(opt)
    register_sim_camera()
    if protocol != "off":
        assert drift is not None, "protocol requires a drift environment"
        install_capture_protocol(protocol, drift)

    before = _artifact_roots(data_root)
    t0 = time.perf_counter()
    exit_code = 0
    try:
        slm_pib_run.main(args=_click_args(data_root, epochs, abba=abba), standalone_mode=True)
    except SystemExit as exc:  # Click 的 standalone 退出
        exit_code = int(exc.code) if isinstance(exc.code, int) else 0
    wall = time.perf_counter() - t0

    pkl = _newest_pkl(data_root, before)
    rows = _load_rows(pkl) if pkl is not None else []
    wrapper = current_wrapper()

    result: dict[str, Any] = {
        "variant": name,
        "protocol": protocol,
        "abba": bool(abba),
        "epochs": int(epochs),
        "exit_code": exit_code,
        "wall_s": round(wall, 1),
        "pkl": str(pkl) if pkl else None,
        "rows": len(rows),
    }
    if drift is not None:
        result["drift"] = {
            "swing_pct": round(drift.swing_pct(), 1),
            "n_reads": drift.n_reads,
            "stray_adu": drift.stray_adu,
        }
    if wrapper is not None:
        result["same_phase"] = wrapper.same_phase_sigma()
        result["n_protocol_captures"] = len(wrapper.captures)
        result["capture_log"] = wrapper.captures
    result.update(compute_drift_verdict_metrics(rows))
    return result


def _reproduce_noise_gates(
    diffs: list[float], *, k: float = NOISE_GATE_K, window: int = NOISE_GATE_WINDOW
) -> list[bool]:
    """离线复算优化器的噪声门判据 (与 ``slm_zernike_pib._noise_gate`` 同构)。

    recorder 的 pkl **不落盘** ``_gate`` 列 (``slm_shaping_runner._DEBUG_SCALAR_KEYS``
    未列入它), 门控零置数不能从行里直读 —— 曾按不存在的键统计, 得到恒 0 的错误值。
    这里按判据本身复算: 对第 i 个更新, sigma 取**前序** diff (deque, maxlen=window)
    的标准差, ``|diff_i| <= k * sigma`` 即零置; sigma 不可用 (不足 2 个样本) 时门开。
    与优化器日志 "N noise-gated updates" 逐变体一致 (实测 29/30/29/29, 共 32 轮)。
    """
    history: deque[float] = deque(maxlen=window)
    out: list[bool] = []
    for d in diffs:
        sigma = (
            float(np.std(np.asarray(history, dtype=np.float64)))
            if len(history) >= 2
            else 0.0
        )
        out.append(
            bool(
                k > 0.0
                and np.isfinite(sigma)
                and sigma > 0.0
                and abs(d) <= k * sigma
            )
        )
        history.append(float(d))
    return out


def compute_drift_verdict_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """从 recorder 行计算全部 issue 判据指标。

    能量护栏 (``--max-energy-loss`` 默认 0.6) 触发的轮次记录的是惩罚值
    (``J ± 1e3``), 不是分数 —— 先按哨兵值剔除并单独计数, 否则 dec/late_gain
    会被 ±1000 的尖峰完全支配。
    """
    if not rows:
        return {"metrics": "no-rows"}
    rows = sorted(rows, key=lambda r: float(r.get("_epoch", 0)))
    j_all = np.array([float(r["J"]) for r in rows], dtype=np.float64)
    max_brt = np.array([float(r["max_brt"]) for r in rows], dtype=np.float64)
    guard_mask = np.abs(j_all) > 100.0  # 护栏哨兵 (|J| 正常量级 ~10)
    scored = ~guard_mask
    j = j_all[scored]
    diffs = np.array(
        [
            abs(float(r.get("_diff", 0.0)))
            for i, r in enumerate(rows[1:], start=1)
            if not guard_mask[i]
        ],
        dtype=np.float64,
    )
    # 噪声门零置数按判据复算 (pkl 无 ``_gate`` 列, 见 ``_reproduce_noise_gates``)。
    # 复算覆盖**全部**更新行 (含护栏行 —— 护栏行的 diff 仍是真实测量值, 会进
    # 优化器的 diff 历史), 只有 dec/late_gain 等指标才剔除哨兵行。
    gate_flags = _reproduce_noise_gates(
        [float(r.get("_diff", 0.0)) for r in rows[1:]]
    )
    n_gate_zeroed = int(sum(gate_flags))

    if j.size < 2:
        return {
            "metrics": "too-few-scored-rows",
            "guard_rows": int(guard_mask.sum()),
            "n_gate_zeroed": n_gate_zeroed,
        }
    stats = analyse_trace(j, delta=0.2, lower_is_better=False)
    r2 = float("nan")
    if max_brt[scored].size > 1 and max_brt[scored].std() > 0 and j.std() > 0:
        r2 = float(np.corrcoef(j, max_brt[scored])[0, 1] ** 2)
    return {
        "J_first": float(j[0]),
        "J_final": float(j[-1]),
        "J_best": float(j.max()),
        "J_span_pct": float(100.0 * (j.max() - j.min()) / abs(j[0])) if j[0] else float("nan"),
        "max_brt_swing_pct": (
            float(100.0 * (max_brt[scored].max() - max_brt[scored].min()) / max_brt[scored].mean())
            if max_brt[scored].mean()
            else float("nan")
        ),
        "r2_J_maxbrt": r2,
        "dec": stats.frac_decreasing,
        "late_gain_pct": stats.late_gain_pct,
        "verdict": stats.verdict(),
        "diff_abs_mean": float(diffs.mean()) if diffs.size else float("nan"),
        "diff_abs_median": float(np.median(diffs)) if diffs.size else float("nan"),
        "gate_fraction": (
            float(np.mean(gate_flags)) if gate_flags else float("nan")
        ),
        "n_gate_zeroed": n_gate_zeroed,
        "guard_rows": int(guard_mask.sum()),
        "scored_rows": int(scored.sum()),
    }


# === SECTION: 报告 =========================================================


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        return format(value, spec)
    return str(value)


def _fig(name: str, stamp: str = "") -> str:
    """图链接 (report/AGENTS.md §5: 同目录资产不带 ``../``; 图名带日期戳)。"""
    return f"figures/{name}_{stamp}.png" if stamp else f"figures/{name}.png"


def write_report(
    results: list[dict[str, Any]],
    out: Path,
    meta: dict[str, Any],
    *,
    intrinsic: dict[str, Any] | None = None,
    fig_stamp: str = "",
) -> Path:
    """把 A/B 矩阵写成 markdown 报告 (判据逐条对照 issue 原文)。

    ``intrinsic`` 是本底 σ 探针 (drift-off 运行) 的 results.json 内容, 用于在
    同相位噪声底表里给出 3σ 判据的基准行 —— 没有它, 各变体的 σ 无法判断有多少
    来自漂移、多少来自孪生本身的读出噪声。
    """
    lines: list[str] = []
    lines.append("# SPGD 闭环照明漂移对消 A/B (数字孪生, #144 第 1 项 / #36 判据)")
    lines.append("")
    lines.append(f"- 日期: {meta['date']}")
    lines.append(f"- 孪生种子: {meta['system_seed']}; 漂移种子: {meta['drift_seed']}")
    lines.append(
        f"- 环境: 平滑漂移 a={meta['a_smooth']} T={meta['t_smooth']} 次读出, "
        f"随机游走 σ={meta['walk_sigma']} 截断 ±{meta['walk_clip']}, "
        f"杂散底板 B={meta['stray_adu']} ADU/px"
    )
    lines.append(
        f"- 运行: {meta['epochs']} epoch/变体, 固定曝光 80 ms, 目标 100x100, "
        "delta=0.2 / lr=0.3 钉住 (调度器不接管)"
    )
    lines.append("")
    lines.append("## 总表")
    lines.append("")
    lines.append(
        "| 变体 | 协议 | dec (>0.55) | late_gain (>=10%) | verdict | "
        "R²(J~亮度) | J 首→末 | 亮度摆幅 | 噪声门零置 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['protocol']}{'+abba' if r['abba'] else ''} "
            f"| {_fmt(r.get('dec'), '.2f')} | {_fmt(r.get('late_gain_pct'), '+.1f')} "
            f"| {r.get('verdict', '-')} | {_fmt(r.get('r2_J_maxbrt'), '.3f')} "
            f"| {_fmt(r.get('J_first'), '+.3f')} → {_fmt(r.get('J_final'), '+.3f')} "
            f"| {_fmt(r.get('max_brt_swing_pct'), '.0f')}% "
            f"| {r.get('n_gate_zeroed', '-')}/{r.get('rows', 0) - 1} |"
        )
    lines.append("")
    lines.append(
        "噪声门零置列为**离线复算** (pkl 不落盘 ``_gate`` 列, 按优化器判据 "
        "``|diff| <= 3σ(前序 20 个 diff)`` 用 ``_diff`` 列重算, 与优化器日志一致)。"
    )
    lines.append("")
    lines.append("![J 轨迹](" + _fig("j_trace", fig_stamp) + ")")
    lines.append("")
    lines.append("![亮度轨迹](" + _fig("max_brt_trace", fig_stamp) + ")")
    lines.append("")
    lines.append("## 同相位噪声底与可分辨性 (#144 测试方法)")
    lines.append("")
    lines.append("| 变体 | σ(J) 同相位 | 3σ 判据 | 协议采集次数 |")
    lines.append("|---|---|---|---|")
    if intrinsic is not None:
        ir = intrinsic["results"][0]
        sp = ir.get("same_phase", {})
        if sp and np.isfinite(sp.get("sigma", float("nan"))):
            lines.append(
                f"| intrinsic (drift off) | {_fmt(sp['sigma'], '.4f')} (n={int(sp['n'])}) "
                f"| 基准行: 漂移关、杂散底板开时的读出+评估噪声底 | "
                f"{ir.get('n_protocol_captures', '-')} |"
            )
    for r in results:
        sp = r.get("same_phase")
        if not sp or not np.isfinite(sp.get("sigma", float("nan"))):
            lines.append(f"| {r['variant']} | (无参考帧记录) | - | - |")
            continue
        thr = 3.0 * sp["sigma"]
        resolvable = r.get("diff_abs_median", float("nan"))
        ok = "是" if np.isfinite(resolvable) and resolvable > thr else "否"
        lines.append(
            f"| {r['variant']} | {_fmt(sp['sigma'], '.4f')} (n={int(sp['n'])}) "
            f"| 3σ={_fmt(thr, '.4f')}, 中位 abs(ΔJ)={_fmt(resolvable, '.4f')} → {ok} "
            f"| {r.get('n_protocol_captures', '-')} |"
        )
    lines.append("")
    lines.append("## 逐变体明细")
    for r in results:
        lines.append("")
        lines.append(f"### {r['variant']} ({r['protocol']}{', abba' if r['abba'] else ''})")
        lines.append("")
        lines.append(f"- 退出码 {r['exit_code']}, 墙钟 {r['wall_s']}s, recorder 行数 {r['rows']}")
        if "drift" in r:
            lines.append(
                f"- 增益摆幅 {r['drift']['swing_pct']}% (共 {r['drift']['n_reads']} 次读出)"
            )
        lines.append(
            f"- dec={_fmt(r.get('dec'), '.3f')}, late_gain={_fmt(r.get('late_gain_pct'), '+.1f')}%, "
            f"verdict={r.get('verdict')}"
        )
        lines.append(
            f"- R²(J~max_brt)={_fmt(r.get('r2_J_maxbrt'), '.3f')}, "
            f"max_brt 摆幅={_fmt(r.get('max_brt_swing_pct'), '.1f')}%"
        )
        lines.append(
            f"- |ΔJ| 均值={_fmt(r.get('diff_abs_mean'), '.4f')}, "
            f"中位={_fmt(r.get('diff_abs_median'), '.4f')}, "
            f"噪声门零置 {r.get('n_gate_zeroed')} 轮"
        )
        log = r.get("capture_log") or []
        pts = [(c["n"], c["g_true"]) for c in log if c.get("g_true") is not None]
        if len(pts) >= 2:
            lines.append("")
            lines.append(
                f"![增益轨迹 {r['variant']}]("
                + _fig(f"drift_gain_{r['variant']}", fig_stamp) + ")"
            )
    lines.append("")
    lines.append("## 解读边界")
    lines.append("")
    lines.append(
        "- 本运行在数字孪生上闭环, 验证的是**协议机制** (漂移对消能否把梯度信噪比 "
        "拉回可分辨区), 不能替代 #144/#36 的实机验收; 实机结论仍需在台架复跑。"
    )
    lines.append(
        "- 孪生 ``SimPibCCD.get_numpy_image`` 忽略 ``n_sample`` (不做多帧平均), "
        "所有 σ 按单帧读出解释; 真实相机的 3 帧平均会进一步压低噪声底。"
    )
    lines.append(
        "- 纯倍增增益对孪生 shape 目标不可见 (各项同缩放); 复现 bench 的亮度耦合 "
        "依赖杂散底板 B (bench 上同款通道: 总能量分母中的背景 / ROI 固定底噪)。"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# === SECTION: 图 ===========================================================


def _variant_color(name: str) -> str:
    return {
        "control": "#1f77b4",
        "baseline": "#d62728",
        "abba": "#ff7f0e",
        "flat_norm": "#2ca02c",
    }.get(name, "#7f7f7f")


def write_figures(
    results: list[dict[str, Any]], out_dir: Path, *, stamp: str = ""
) -> list[Path]:
    """从 results.json + recorder pkl 画 A/B 对比图 (J 轨迹 / 亮度轨迹 / 增益轨迹)。

    ``stamp`` 非空时图名带日期戳 (report/AGENTS.md §2.1: 同系列每份报告独占图)。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in results:
        pkl = r.get("pkl")
        if not pkl or not Path(pkl).is_file():
            continue
        rows = _load_rows(Path(pkl))
        rows = sorted(rows, key=lambda x: float(x.get("_epoch", 0)))
        if not rows:
            continue
        epochs = np.array([float(x.get("_epoch", 0)) for x in rows], dtype=np.float64)
        j = np.array([float(x["J"]) for x in rows], dtype=np.float64)
        series[r["variant"]] = (epochs, j)

    if series:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, (ep, j) in series.items():
            ax.plot(ep, j, marker="o", ms=3, lw=1.2, color=_variant_color(name), label=name)
        ax.set_xlabel("epoch")
        ax.set_ylabel("J (shape score, higher is better)")
        ax.set_title("SPGD closed loop: objective trace per protocol (twin)")
        ax.legend()
        ax.grid(alpha=0.3)
        path = out_dir / f"j_trace_{stamp}.png" if stamp else out_dir / "j_trace.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for r in results:
            pkl = r.get("pkl")
            if not pkl or not Path(pkl).is_file():
                continue
            rows = sorted(_load_rows(Path(pkl)), key=lambda x: float(x.get("_epoch", 0)))
            if not rows:
                continue
            ep = [float(x.get("_epoch", 0)) for x in rows]
            brt = [float(x["max_brt"]) for x in rows]
            ax.plot(ep, brt, marker="o", ms=3, lw=1.2, color=_variant_color(r["variant"]),
                    label=r["variant"])
        ax.set_xlabel("epoch")
        ax.set_ylabel("max_brt (camera counts)")
        ax.set_title("Measured brightness per epoch (drift visibility + normalization)")
        ax.legend()
        ax.grid(alpha=0.3)
        path = (
            out_dir / f"max_brt_trace_{stamp}.png"
            if stamp
            else out_dir / "max_brt_trace.png"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 增益轨迹: 协议日志里候选读出的真实增益 (每个候选一个点)。
    for r in results:
        log = r.get("capture_log") or []
        pts = [(c["n"], c["g_true"]) for c in log if c.get("g_true") is not None]
        if len(pts) < 2:
            continue
        ns = np.array([p[0] for p in pts], dtype=np.float64)
        gs = np.array([p[1] for p in pts], dtype=np.float64)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(ns, gs, marker="o", ms=3, lw=1.0, color=_variant_color(r["variant"]))
        ax.set_xlabel("camera read index")
        ax.set_ylabel("laser gain g(t)")
        ax.set_title(f"Illumination drift seen at candidate reads ({r['variant']})")
        ax.grid(alpha=0.3)
        path = (
            out_dir / f"drift_gain_{r['variant']}_{stamp}.png"
            if stamp
            else out_dir / f"drift_gain_{r['variant']}.png"
        )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


# === SECTION: CLI ==========================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 harness 命令行。"""
    parser = argparse.ArgumentParser(
        description="SPGD 闭环照明漂移对消 A/B (数字孪生, #144/#36)"
    )
    parser.add_argument("--epochs", type=int, default=32, help="每变体 SPGD 轮数 (default: 32)")
    parser.add_argument("--smoke", action="store_true", help="2 epoch 接线自检后退出")
    parser.add_argument(
        "--variants",
        default="control,baseline,abba,flat_norm",
        help="逗号分隔的变体列表 (default: 全部四个)",
    )
    parser.add_argument("--a-smooth", type=float, default=DEFAULT_A_SMOOTH)
    parser.add_argument("--t-smooth", type=float, default=DEFAULT_T_SMOOTH)
    parser.add_argument("--walk-sigma", type=float, default=DEFAULT_WALK_SIGMA)
    parser.add_argument("--walk-clip", type=float, default=DEFAULT_WALK_CLIP)
    parser.add_argument("--stray-adu", type=float, default=DEFAULT_B_PX)
    parser.add_argument("--drift-seed", type=int, default=DRIFT_SEED)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="运行产物根目录 (仓库外; 默认 = 系统临时目录下的 ao_spgd_drift_scratch)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="报告输出路径 (default: <data-root>/report.md)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="不跑闭环, 只从 <data-root>/results.json + pkl 重新生成图",
    )
    parser.add_argument(
        "--intrinsic",
        type=Path,
        default=None,
        help=(
            "本底 σ 探针 (drift-off 运行) 的 results.json; 给出后, 报告的同相位"
            "噪声底表增加 intrinsic 基准行, 并复制到 artifacts/"
        ),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help=(
            "不跑闭环: 从 <data-root>/results.json 里各变体的 pkl 重算指标"
            " (修 compute_drift_verdict_metrics 的 bug 后用), 重写双落点报告与图"
        ),
    )
    return parser.parse_args(argv)


def _make_drift(args: argparse.Namespace) -> IlluminationDrift:
    return IlluminationDrift(
        a_smooth=args.a_smooth,
        t_smooth=args.t_smooth,
        walk_sigma=args.walk_sigma,
        walk_clip=args.walk_clip,
        stray_adu=args.stray_adu,
        seed=args.drift_seed,
    )


def _emit_outputs(
    results: list[dict[str, Any]],
    meta: dict[str, Any],
    data_root: Path,
    intrinsic: dict[str, Any] | None,
    intrinsic_path: Path | None,
    out: Path | None = None,
    repo: bool = True,
) -> None:
    """双落点写报告+图, 并把 results.json 复制进仓库 artifacts/ (输入契约)。

    ``repo=False`` (smoke 自检) 只写仓库外速览报告 —— 2 epoch 的接线数字没有
    结论价值, 不应覆盖仓库正式报告。
    """
    scratch_report = out or (data_root / "report.md")
    write_report(results, scratch_report, meta, intrinsic=intrinsic)
    (data_root / "results.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_figures(results, data_root / "figures")
    if not repo:
        logger.info("smoke mode: scratch report only ({})", scratch_report)
        return
    repo_report = write_report(
        results,
        OUT_DIR / Path(REPORT_KEY).name,
        meta,
        intrinsic=intrinsic,
        fig_stamp=FIG_STAMP,
    )
    written = write_figures(results, FIG_DIR, stamp=FIG_STAMP)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(data_root / "results.json", ARTIFACT_DIR / "results.json")
    if intrinsic is not None and intrinsic_path is not None:
        shutil.copy(intrinsic_path, ARTIFACT_DIR / "intrinsic_sigma_probe.json")
    logger.info("report written: {} ({} figures)", repo_report, len(written))


def main() -> None:
    """Entry point: 跑 A/B 矩阵并写报告 (或仅重画图 / 重算指标)。

    报告双落点: ``<data-root>/report.md`` (仓库外速览, 无戳图) 与
    ``report/shape_trials/spgd_drift_cancel_loop/`` (仓库正式报告, 图带日期戳,
    ``artifacts/results.json`` 作为输入契约) —— 后者路径常量写死在
    :data:`REPORT_KEY` (report/AGENTS.md §3)。
    """
    args = parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        payload = json.loads((args.data_root / "results.json").read_text(encoding="utf-8"))
        written = write_figures(payload["results"], args.data_root / "figures")
        written += write_figures(payload["results"], FIG_DIR, stamp=FIG_STAMP)
        logger.info("figures written: {}", [str(p) for p in written])
        return

    if args.recompute:
        payload = json.loads(
            (args.data_root / "results.json").read_text(encoding="utf-8")
        )
        for r in payload["results"]:
            pkl = r.get("pkl")
            if not pkl or not Path(pkl).is_file():
                logger.warning("pkl missing for {}: {}", r["variant"], pkl)
                continue
            metrics = compute_drift_verdict_metrics(_load_rows(Path(pkl)))
            r.update(metrics)  # dec/late_gain/r2/摆幅/diff 统计/门控/J_*/护栏行
            logger.info(
                "{} recomputed: dec={:.3f} gate_zeroed={}",
                r["variant"],
                metrics.get("dec", float("nan")),
                metrics.get("n_gate_zeroed"),
            )
        intrinsic = None
        if args.intrinsic is not None and args.intrinsic.is_file():
            intrinsic = json.loads(args.intrinsic.read_text(encoding="utf-8"))
        _emit_outputs(
            payload["results"], payload["meta"], args.data_root, intrinsic,
            args.intrinsic, args.out,
        )
        return

    epochs = 2 if args.smoke else args.epochs
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    plan: list[tuple[str, bool, IlluminationDrift | None, str, bool]] = [
        # (变体, 有漂移, 漂移模型, 协议, abba)
        ("control", False, None, "off", False),
        ("baseline", True, None, "observe", False),
        ("abba", True, None, "observe", True),
        ("flat_norm", True, None, "normalize", False),
    ]

    results: list[dict[str, Any]] = []
    for name, has_drift, _d, protocol, abba in plan:
        if name not in variants:
            continue
        drift = _make_drift(args) if has_drift else None
        logger.info(
            "=== variant {} (protocol={}, abba={}, drift={}) ===",
            name,
            protocol,
            abba,
            "on" if drift else "off",
        )
        result = run_variant(
            name,
            epochs=epochs,
            drift=drift,
            protocol=protocol,
            abba=abba,
            data_root=args.data_root,
        )
        results.append(result)
        logger.info(
            "variant {} done: dec={:.3f} late_gain={:+.1f}% verdict={} "
            "r2={:.3f} gate_zeroed={} (wall {:.0f}s)",
            name,
            result.get("dec", float("nan")),
            result.get("late_gain_pct", float("nan")),
            result.get("verdict"),
            result.get("r2_J_maxbrt", float("nan")),
            result.get("n_gate_zeroed"),
            result["wall_s"],
        )

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "system_seed": SYSTEM_SEED,
        "drift_seed": args.drift_seed,
        "a_smooth": args.a_smooth,
        "t_smooth": args.t_smooth,
        "walk_sigma": args.walk_sigma,
        "walk_clip": args.walk_clip,
        "stray_adu": args.stray_adu,
        "epochs": epochs,
    }
    intrinsic = None
    if args.intrinsic is not None and args.intrinsic.is_file():
        intrinsic = json.loads(args.intrinsic.read_text(encoding="utf-8"))
    _emit_outputs(
        results, meta, args.data_root, intrinsic, args.intrinsic, args.out,
        repo=not args.smoke,
    )


if __name__ == "__main__":
    main()
