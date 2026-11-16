"""``hw_forward_model_fidelity_ladder.py`` 的纯数值分段 (零设备导入)。

本模块承担保真度阶梯的全部**离线可演练**逻辑: 图样构建、模型远场预测、
实测帧重采样到模型网格 (含零填充覆盖掩膜)、逐图样一致性指标 (Spearman /
Pearson / 径向相关 / FWHM 比 / 峰值偏移)、``fidlad_*`` 图与 metrics/summary
落盘。``--dry-run`` 与 ``--execute`` 两条路径共用这里同一套函数 —— 硬件
会话上跑的代码就是离线演练过的代码, 口径不因路径而分叉。

🔴 本模块**不 import 任何** ``ao_shaping.drivers.*``; 主脚本在收尾用
``assert_no_drivers`` 把这条约束变成可执行的保证。

复用清单 (相对 ``scripts/slm/hw_differentiable_phase_opt.py``, 经
:func:`load_opt_module` 以文件路径加载并注册进 ``sys.modules``; 行号以
2026-11-16 该文件 3714 行的版本为准, 改动了那边请同步这里):

* ``ModelSpec`` (L319) / ``build_model_spec`` (L408): 模型网格;
* ``build_source_amplitude`` (L444): 高斯照明 × 硬口径源振幅;
* ``build_target`` (L466): gs_init 图样的目标远场;
* ``build_initial_phase`` (L2098): GS 交替投影初值;
* ``phase_model_to_panel`` (L537): 模型网格相位 → 面板相位;
* ``resample_ccd_to_model`` (L576): 实测 CCD 帧 → 模型远场网格
  (经验标度 ``scale = K/(M·dp)``, 越界**零填充**);
* ``compare_farfields`` (L608) + ``_AXIS_CANDIDATES`` (L202):
  8 种面板↔相机轴向口径按强度相关择优 (本模块 :func:`sweep_axis_convention`
  复刻其主循环);
* ``model_far_field`` (L3294): ``fftshift(fft2(A·e^{iφ}))`` 远场预测;
* ``assert_no_drivers`` (L895): dry-run 收尾的「零驱动导入」硬保证;
* ``prepare_frame`` / ``measure_spot`` (opt L163-166 处 import, 零导入叶子);
* ``capture_frames`` (L994) / ``save_ccd_frame`` (L983) /
  ``plot_model_vs_hardware`` (L1189) / ``write_csv`` (L950) /
  ``judge_focus_gate`` (L3303) / ``plot_focus_sweep`` (L1016) /
  ``guarded_device`` (L961): 由主脚本在对应路径调用。

最小复刻 (注明来源, 避免为几行代码牵入整个模块):

* :func:`clock` / :func:`sha256_file` ← ``shape_transition_timing_20261106.py``
  L76-86 (该模块的 module-level import 会牵入优化器族, 只为两个 5 行小工具
  不值得);
* :func:`apply_axis` ← ``hw_differentiable_phase_opt._apply_axis_convention``
  L1177 (去掉 float 强转, 让布尔覆盖掩膜也能走同一变换)。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
for _p in (
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(Path(__file__).resolve().parent),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前

import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.stats import pearsonr, spearmanr

from ao_shaping.utils.wavefront.zernike_calc import generate_noll_polynomial

#: 被复用的主优化脚本 (以文件路径加载, 见 :func:`load_opt_module`)。
OPT_SCRIPT_PATH = ROOT / "scripts" / "slm" / "hw_differentiable_phase_opt.py"
OPT_MODULE_NAME = "hw_differentiable_phase_opt"

#: 内置阶梯图样 (顺序即上屏顺序; flat 放最前 —— 硬件路径用同一帧兼做定心)。
BUILTIN_PATTERN_IDS: tuple[str, ...] = (
    "flat",
    "gs_init",
    "optimized_final",
    "defocus_z4_pos",
    "defocus_z4_neg",
    "tilt_x",
    "tilt_y",
)

#: 内置 Zernike 图样 → (Noll n, Noll m, 符号)。与
#: ``absolute_fidelity_probe_20261116.py`` 同一套 Noll 约定:
#: tilt = (1,±1), defocus = (2,0)。
NOLL_SPECS: dict[str, tuple[int, int, float]] = {
    "defocus_z4_pos": (2, 0, +1.0),
    "defocus_z4_neg": (2, 0, -1.0),
    "tilt_x": (1, 1, +1.0),
    "tilt_y": (1, -1, +1.0),
}

#: dry-run 替身帧映射: 阶梯图样 id → ``absolute_fidelity_12bit_fixed_20261116``
#: 里的 ``frames/<id>.npy`` / ``phase/<id>.npy`` 名。缺替身的图样走
#: :func:`synthetic_ccd_frame` 合成帧, 保证整条 metrics/figures 流水线都被演练。
STANDIN_MAP: dict[str, str] = {
    "flat": "zero",
    "defocus_z4_pos": "defocus_z4_1rad",
    "tilt_x": "tilt_x_1rad",
    "tilt_y": "tilt_y_1rad",
}

#: 合成替身帧的画布 (行, 列) —— MER2-507-23GM 全幅。
SYNTH_FRAME_SHAPE: tuple[int, int] = (1944, 2592)
#: 合成帧的峰值计数 / 背景 / 噪声 (对齐 12-bit 替身帧的量级: 峰值 ~2000)。
SYNTH_PEAK_COUNTS = 2000.0
SYNTH_BACKGROUND = 2.0
SYNTH_NOISE_STD = 6.0


def load_opt_module() -> Any:
    """按文件路径加载 ``hw_differentiable_phase_opt.py`` 并注册进 sys.modules。

    注册必须在 ``exec_module`` **之前** —— 该脚本内部 import
    ``scripts.*`` 子模块, 若自身不在 ``sys.modules`` 里, 二次加载会重复执行
    module-level 代码。返回的模块只含零导入叶子 (实测不牵入
    ``ao_shaping.drivers.*``), dry-run 加载它是安全的。
    """
    if OPT_MODULE_NAME in sys.modules:
        return sys.modules[OPT_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(OPT_MODULE_NAME, OPT_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法按路径加载 {OPT_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clock() -> dict[str, Any]:
    """device_log 条目的时间戳字段 (复刻 shape_transition_timing_20261106.py L76-80)。"""
    return dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        monotonic_ns=time.monotonic_ns(),
    )


def sha256_file(path: Path) -> str | None:
    """文件 sha256 (复刻 shape_transition_timing_20261106.py L83-86)。"""
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 几何 / 模型网格
# ---------------------------------------------------------------------------
def load_geometry_npz(path: Path) -> dict[str, Any]:
    """读几何标定 npz, 给出各向同性远场标度 ``k_mean`` 与标定的相机中心。

    ``center`` 键的约定与 ``hw_differentiable_phase_opt.py`` L2694 一致:
    ``(centre_cy, centre_cx)`` = (行, 列)。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"几何标定文件不存在: {p}")
    d = np.load(p)
    kx = float(d["Kx"])
    ky = float(d["Ky"])
    centre_cy, centre_cx = (float(v) for v in d["center"])
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "Kx": kx,
        "Ky": ky,
        "k_mean": 0.5 * (kx + ky),
        "rotation_deg": float(d["rotation_deg"]),
        "fwhm0_px": float(d["fwhm0"]),
        "mirror": bool(d["mirror"]),
        "crop_side": int(d["crop_side"]),
        "centre_xy": (centre_cx, centre_cy),
    }


def coverage_mask(
    *,
    frame_shape: tuple[int, int],
    centre_xy: tuple[float, float],
    k_mean: float,
    spec: Any,
) -> np.ndarray:
    """模型远场网格上「被传感器覆盖」的布尔掩膜 (零填充的补集)。

    :func:`resample_ccd_to_model` (opt L520-551) 把越出传感器边界的格填成 0 ——
    这些格里的 0 是**外推**不是测量, 混进相关系数/径向剖面会稀释真实一致性。
    本掩膜用与它完全相同的坐标公式标出 in-bounds 的格 (旋转取 0°, 与其调用
    口径一致; 面板↔相机的轴向差异由 :func:`sweep_axis_convention` 的
    转置/镜像择优处理, 那些变换不改变覆盖集合的形状)。
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    m = int(spec.m_grid)
    scale = spec.far_field_sample_ccd_px(k_mean)
    span = scale * m
    offs = np.arange(m, dtype=np.float64) * scale - span / 2.0
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    cols = cx + offs
    rows = cy + offs
    in_cols = (cols >= -0.5) & (cols <= width - 0.5)
    in_rows = (rows >= -0.5) & (rows <= height - 0.5)
    return np.broadcast_to(in_rows[:, None] & in_cols[None, :], (m, m)).copy()


def resample_with_coverage(
    opt_mod: Any,
    frame: np.ndarray,
    centre_xy: tuple[float, float],
    k_mean: float,
    spec: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """实测 CCD 帧 → (模型网格上的实测远场, 覆盖掩膜)。

    重采样本体复用 opt ``resample_ccd_to_model`` (L520-551), 帧预处理复用
    ``prepare_frame`` (opt L2953-2965 的同款调用链)。
    """
    prepared = opt_mod.prepare_frame(np.asarray(frame, dtype=np.float64))
    meas = opt_mod.resample_ccd_to_model(prepared, centre_xy, k_mean, spec)
    cov = coverage_mask(
        frame_shape=(int(prepared.shape[0]), int(prepared.shape[1])),
        centre_xy=centre_xy,
        k_mean=k_mean,
        spec=spec,
    )
    return np.asarray(meas, dtype=np.float64), cov


def apply_axis(arr: np.ndarray, cmp_metrics: dict[str, Any]) -> np.ndarray:
    """按择优轴向口径变换数组 (复刻 opt ``_apply_axis_convention`` L1121-1131)。

    不强转 dtype —— 覆盖掩膜 (bool) 与强度 (float) 走同一个变换。
    """
    out = np.asarray(arr)
    if cmp_metrics.get("transpose"):
        out = out.T
    if cmp_metrics.get("flip_rows"):
        out = out[::-1, :]
    if cmp_metrics.get("flip_cols"):
        out = out[:, ::-1]
    return out


def sweep_axis_convention(
    opt_mod: Any,
    model: np.ndarray,
    measured: np.ndarray,
) -> dict[str, Any]:
    """8 种面板↔相机轴向口径全试一遍, 按强度相关择优 (复刻 opt L2972-2998)。

    Returns:
        最优口径的 ``compare_farfields`` 结果, 附 ``transpose`` / ``flip_rows``
        / ``flip_cols`` / ``orientation_match`` 键。
    """
    candidates: dict[str, dict[str, Any]] = {}
    for name, transpose, flip_rows, flip_cols in opt_mod._AXIS_CANDIDATES:
        variant = np.asarray(measured, dtype=np.float64)
        if transpose:
            variant = variant.T
        if flip_rows:
            variant = variant[::-1, :]
        if flip_cols:
            variant = variant[:, ::-1]
        candidates[name] = {
            **opt_mod.compare_farfields(model, variant, transpose_measured=False),
            "transpose": bool(transpose),
            "flip_rows": bool(flip_rows),
            "flip_cols": bool(flip_cols),
        }
    best_key = max(
        candidates,
        key=lambda key: (
            candidates[key]["intensity_correlation"]
            if np.isfinite(candidates[key]["intensity_correlation"])
            else -np.inf
        ),
    )
    best = candidates[best_key]
    best["orientation_match"] = best_key
    return best


# ---------------------------------------------------------------------------
# 图样构建
# ---------------------------------------------------------------------------
@dataclass
class Pattern:
    """阶梯里的一张相位图样。

    ``phase_block`` 是**模型瞳面块** ``(N, N)`` 弧度 (单位圆外为 0 的 Zernike
    直接生成; GS/自定义图样也归一到这个口径), 由 :func:`embed_block` 放进
    ``M×M`` 模型网格后再做预测与上面板。
    """

    pattern_id: str
    source: str
    note: str
    phase_block: np.ndarray
    standin_id: str | None = None


def embed_block(spec: Any, block: np.ndarray) -> np.ndarray:
    """把 ``(N, N)`` 瞳面块嵌进 ``M×M`` 模型网格 (块外补零 = 零填充)。"""
    arr = np.asarray(block, dtype=np.float64)
    m = int(spec.m_grid)
    n = int(spec.n_pupil)
    if arr.shape == (m, m):
        return arr.copy()
    if arr.shape != (n, n):
        raise ValueError(
            f"相位形状 {arr.shape} 既不是瞳面块 {(n, n)} 也不是模型网格 {(m, m)}"
        )
    out = np.zeros((m, m), dtype=np.float64)
    start = int(spec.pupil_start)
    out[start : start + n, start : start + n] = arr
    return out


def panel_to_block(panel: np.ndarray, spec: Any) -> np.ndarray:
    """从面板相位里抠回 ``(N, N)`` 瞳面块 (``phase_model_to_panel`` 的逆变换)。

    块索引 ``j`` 对应模型索引 ``pupil_start + j``, 面板坐标
    ``x = cx + (mj - block_centre)·dp`` (最近邻取整, 与正向映射同口径)。
    块外的面板相位**不进模型** —— 模型口径只覆盖瞳面块, 报告里会写明。
    """
    arr = np.asarray(panel, dtype=np.float64)
    h, w = arr.shape
    cx, cy = spec.pupil_center
    mj = int(spec.pupil_start) + np.arange(int(spec.n_pupil))
    off = (mj - spec.block_centre) * spec.dp
    xs = np.clip(np.rint(cx + off).astype(np.int64), 0, w - 1)
    ys = np.clip(np.rint(cy + off).astype(np.int64), 0, h - 1)
    return arr[np.ix_(ys, xs)]


def resolve_optimized_phase(
    explicit: str | Path | None,
    search_root: Path,
) -> tuple[Path | None, str]:
    """定位 ``optimized_final`` 图样的相位 npy。

    优先显式路径; 否则在 ``search_root`` (默认
    ``data/debug/differentiable_opt``) 下按修改时间取最新的
    ``phase_model.npy``, 其次 ``phase_final.npy``。找不到返回 ``(None, 说明)``
    —— 阶梯**跳过**该图样而不是报错: 它依赖之前的优化运行是否留下产物。
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"--optimized-phase 不存在: {p}")
        return p, f"explicit:{p}"
    for name in ("phase_model.npy", "phase_final.npy"):
        hits = sorted(
            Path(search_root).glob(f"*/{name}"),
            key=lambda q: q.stat().st_mtime,
        )
        if hits:
            return hits[-1], f"latest:{hits[-1]}"
    return None, f"未在 {search_root} 找到 phase_model.npy / phase_final.npy"


def noll_phase_block(zn: int, zm: int, n: int, amp: float) -> np.ndarray:
    """单个 Noll 图样的 ``(N, N)`` 瞳面块 (``amp`` = 圆内 RMS 弧度)。

    :func:`generate_noll_polynomial` 在单位圆外返回 ``NaN``; SLM 上圆外命令 0
    相位, 故显式置 0 —— 与 ``absolute_fidelity_probe_20261116.noll_phase``
    (该文件 L172-182) 同口径, 约定也写进图样 note / plan.json。
    """
    raw = generate_noll_polynomial(int(zn), int(zm), (int(n), int(n)), float(amp))
    return np.nan_to_num(
        np.asarray(raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )


def _clean_phase(arr: np.ndarray, path: Path) -> np.ndarray:
    """把载入相位里的 NaN/Inf 置 0 (与 ``load_anchored_phase`` 同口径), 有则记录。"""
    a = np.asarray(arr, dtype=np.float64)
    bad = int(np.count_nonzero(~np.isfinite(a)))
    if bad:
        logger.warning("{}: 相位含 {} 个非有限值, 已置 0 (SLM 圆外命令 0 相位)", path, bad)
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def resample_model_grid(src: np.ndarray, spec: Any) -> np.ndarray:
    """把**别的网格口径** ``(M', M')`` 的模型相位重采样到本次 ``(M, M)``。

    锚点是网格**几何中心** ``(M-1)/2`` —— :func:`build_model_spec` 里
    ``M = padding·N``, 于是 ``block_centre = (M-1)/2`` 与 ``N`` 无关, 只由
    ``M`` 决定。两块网格的物理跨度都是 ``padding·span`` 面板像元, 所以中心
    锚定的纯缩放 ``i = (M-1)/2 + (i'-(M'-1)/2)·M/M'`` 就是同一物理面的换采样
    (最近邻, 不插值出新相位值)。

    ⚠️ 前提是两边 ``--model-padding`` 与 ``--pupil-radius`` 同默认值 (跨度都是
    960 面板像元); 调用方必须把「重采样过」写进图样说明, 报告里也要有 caveat。
    """
    arr = np.asarray(src, dtype=np.float64)
    m_src = int(arr.shape[0])
    m_dst = int(spec.m_grid)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"模型网格重采样只接受方形数组, 得到 {arr.shape}")
    if m_src == m_dst:
        return arr.copy()
    idx_dst = np.arange(m_dst, dtype=np.float64)
    src_idx = (m_src - 1) / 2.0 + (idx_dst - (m_dst - 1) / 2.0) * (m_src / m_dst)
    take = np.clip(np.rint(src_idx).astype(np.int64), 0, m_src - 1)
    return arr[np.ix_(take, take)]


def _load_custom_phase(
    path: Path,
    spec: Any,
    panel_shape: tuple[int, int],
    *,
    allow_resample: bool = True,
) -> tuple[np.ndarray, str]:
    """加载自定义相位 npy → ``((N, N) 瞳面块, 说明)``。

    接受四种形状: ``(N, N)`` 瞳面块、``(M, M)`` 模型网格 (抠中央块)、
    面板形状 (经 :func:`panel_to_block` 抠块)、以及**别的网格口径**的方形
    ``(M', M')`` 模型相位 (经 :func:`resample_model_grid` 换采样, 说明里会写明
    来源尺寸; ``allow_resample=False`` 时改为抛错)。
    """
    arr = _clean_phase(np.load(path), path)
    n = int(spec.n_pupil)
    m = int(spec.m_grid)
    start = int(spec.pupil_start)
    if arr.shape == (n, n):
        return arr.copy(), ""
    if arr.shape == (m, m):
        return arr[start : start + n, start : start + n].copy(), ""
    if arr.shape == tuple(panel_shape):
        return panel_to_block(arr, spec), ""
    if (
        allow_resample
        and arr.ndim == 2
        and arr.shape[0] == arr.shape[1]
        and arr.shape[0] > n
    ):
        full = resample_model_grid(arr, spec)
        note = (
            f"源网格 {arr.shape[0]}×{arr.shape[1]} → 本次 {m}×{m} "
            "(中心锚定最近邻换采样; 前提: 两边 padding/瞳面半径同默认)"
        )
        return full[start : start + n, start : start + n].copy(), note
    raise ValueError(
        f"{path}: 相位形状 {arr.shape} 不是瞳面块 {(n, n)}、模型网格 {(m, m)}"
        f" 或面板 {tuple(panel_shape)}"
    )


def parse_pattern_tokens(
    raw: str,
    custom_phase_dir: str | Path | None = None,
) -> list[tuple[str, Any]]:
    """解析 ``--patterns`` 逗号列表 → token 序列。

    token 形态: 内置名 (``flat`` 等) / ``all`` / ``name=path.npy`` /
    裸 ``path.npy`` / ``dir:路径`` (目录下全部 ``*.npy``)。
    ``--custom-phase-dir`` 等价于追加一个 ``dir:`` token。
    """
    tokens: list[tuple[str, Any]] = []
    for raw_tok in str(raw).split(","):
        tok = raw_tok.strip()
        if not tok:
            continue
        if tok == "all":
            tokens.extend(("builtin", pid) for pid in BUILTIN_PATTERN_IDS)
        elif tok.startswith("dir:"):
            tokens.append(("dir", Path(tok[4:])))
        elif "=" in tok:
            name, _, path = tok.partition("=")
            tokens.append(("file", (name.strip(), Path(path.strip()))))
        elif tok.endswith(".npy"):
            p = Path(tok)
            tokens.append(("file", (p.stem, p)))
        elif tok in BUILTIN_PATTERN_IDS:
            tokens.append(("builtin", tok))
        else:
            raise ValueError(
                f"未知图样 {tok!r}; 内置: {', '.join(BUILTIN_PATTERN_IDS)},"
                " 或给 name=path.npy / dir:路径"
            )
    if custom_phase_dir:
        tokens.append(("dir", Path(custom_phase_dir)))
    if not tokens:
        raise ValueError("--patterns 解析后为空")
    return tokens


def build_patterns(
    *,
    tokens: list[tuple[str, Any]],
    opt_mod: Any,
    spec: Any,
    amplitude: np.ndarray,
    target: np.ndarray,
    zernike_amp: float,
    gs_iters: int,
    seed: int,
    optimized_phase: str | Path | None,
    optimized_search_root: Path,
    panel_shape: tuple[int, int],
    allow_resample: bool = True,
) -> tuple[list[Pattern], list[dict[str, str]]]:
    """按 token 序列物化阶梯图样。

    Returns:
        ``(patterns, skipped)``; ``skipped`` 记录被跳过图样的
        ``{"pattern_id", "reason"}`` (``optimized_final`` 找不到历史优化产物、
        或产物网格口径对不上且 ``allow_resample=False`` 时会跳过)。
    """
    n = int(spec.n_pupil)
    patterns: list[Pattern] = []
    skipped: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(pat: Pattern) -> None:
        if pat.pattern_id in seen:
            raise ValueError(f"图样 id 重复: {pat.pattern_id}")
        seen.add(pat.pattern_id)
        patterns.append(pat)

    for kind, payload in tokens:
        if kind == "builtin":
            pid = str(payload)
            if pid == "flat":
                add(
                    Pattern(
                        pattern_id="flat",
                        source="builtin:flat",
                        note="零相位基准 (硬件路径同一帧兼做定心)",
                        phase_block=np.zeros((n, n), dtype=np.float64),
                        standin_id=STANDIN_MAP.get("flat"),
                    )
                )
            elif pid == "gs_init":
                phase_full, info = opt_mod.build_initial_phase(
                    spec,
                    amplitude,
                    target,
                    kind="gs",
                    gs_iters=int(gs_iters),
                    seed=int(seed),
                )
                if phase_full is None:  # pragma: no cover - kind="gs" 必返回相位
                    raise RuntimeError("build_initial_phase 未返回 GS 相位")
                start = int(spec.pupil_start)
                add(
                    Pattern(
                        pattern_id="gs_init",
                        source=f"builtin:gs (iters={gs_iters}, seed={seed})",
                        note=(
                            "GS 交替投影初值 (方块目标); converged="
                            f"{info.get('gs_converged')}"
                        ),
                        phase_block=np.asarray(phase_full, dtype=np.float64)[
                            start : start + n, start : start + n
                        ].copy(),
                    )
                )
            elif pid == "optimized_final":
                path, how = resolve_optimized_phase(optimized_phase, optimized_search_root)
                if path is None:
                    skipped.append({"pattern_id": pid, "reason": how})
                    continue
                if not path.is_file():
                    skipped.append(
                        {
                            "pattern_id": pid,
                            "reason": f"解析到的相位已不存在: {path}",
                        }
                    )
                    continue
                try:
                    block, rs_note = _load_custom_phase(
                        path, spec, panel_shape, allow_resample=allow_resample
                    )
                except ValueError as exc:
                    skipped.append({"pattern_id": pid, "reason": str(exc)})
                    continue
                add(
                    Pattern(
                        pattern_id="optimized_final",
                        source=f"builtin:optimized_final ({how})",
                        note="历史可微优化运行的最终相位"
                        + (f"; {rs_note}" if rs_note else ""),
                        phase_block=block,
                    )
                )
            else:
                zn, zm, sign = NOLL_SPECS[pid]
                amp = float(sign) * float(zernike_amp)
                add(
                    Pattern(
                        pattern_id=pid,
                        source=f"builtin:noll({zn},{zm}) amp={amp:+.3g} rad",
                        note=(
                            "单位圆外为 0 (generate_noll_polynomial 圆外 NaN → 0, "
                            "SLM 圆外命令 0 相位); 圆内 RMS = |amp| (Noll 归一)"
                        ),
                        phase_block=noll_phase_block(zn, zm, n, amp),
                        standin_id=STANDIN_MAP.get(pid),
                    )
                )
        elif kind == "file":
            name, path = payload
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f"自定义相位不存在: {path}")
            block, rs_note = _load_custom_phase(
                path, spec, panel_shape, allow_resample=allow_resample
            )
            add(
                Pattern(
                    pattern_id=str(name),
                    source=f"file:{path}",
                    note="自定义相位 npy" + (f"; {rs_note}" if rs_note else ""),
                    phase_block=block,
                )
            )
        elif kind == "dir":
            d = Path(payload)
            if not d.is_dir():
                raise FileNotFoundError(f"自定义相位目录不存在: {d}")
            for path in sorted(d.glob("*.npy")):
                block, rs_note = _load_custom_phase(
                    path, spec, panel_shape, allow_resample=allow_resample
                )
                add(
                    Pattern(
                        pattern_id=path.stem,
                        source=f"dir:{d}",
                        note="自定义相位 npy (目录扫描)"
                        + (f"; {rs_note}" if rs_note else ""),
                        phase_block=block,
                        standin_id=STANDIN_MAP.get(path.stem),
                    )
                )
        else:  # pragma: no cover - parse 已挡住
            raise ValueError(f"未知 token 类型: {kind}")
    return patterns, skipped


# ---------------------------------------------------------------------------
# 合成替身帧 (dry-run 用, 不开任何设备)
# ---------------------------------------------------------------------------
def synthetic_ccd_frame(
    pred: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    centre_xy: tuple[float, float],
    k_mean: float,
    spec: Any,
    seed: int = 0,
    peak_counts: float = SYNTH_PEAK_COUNTS,
    background: float = SYNTH_BACKGROUND,
    noise_std: float = SYNTH_NOISE_STD,
) -> np.ndarray:
    """由模型预测**反向**合成一张全幅 CCD 替身帧 (纯数值, 零设备)。

    坐标公式是 :func:`resample_ccd_to_model` 的逆: 画布像素 → 模型格
    (``map_coordinates`` 线性插值, 画布外为 0), 峰值归一后缩到
    ``peak_counts``, 加背景与高斯噪声 —— 量级对齐 12-bit 替身帧 (峰值 ~2000)。
    用它演练「重采样 → 指标 → 图」整条链: 合成帧回读后与预测应高度相关,
    这正是 metrics 自检的判据。
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    scale = spec.far_field_sample_ccd_px(k_mean)
    c = int(spec.centre_index)
    rows = np.arange(h, dtype=np.float64)
    cols = np.arange(w, dtype=np.float64)
    dy = (rows - float(centre_xy[1]))[:, None]
    dx = (cols - float(centre_xy[0]))[None, :]
    mi = c + dy / scale  # 旋转取 0°, 与 resample_ccd_to_model 的调用口径一致
    mj = c + dx / scale
    coords = np.broadcast_arrays(mi, mj)
    canvas = map_coordinates(
        np.asarray(pred, dtype=np.float64),
        np.stack(coords, axis=0),
        order=1,
        mode="constant",
        cval=0.0,
    )
    peak = float(canvas.max())
    if peak > 0:
        canvas = canvas / peak * float(peak_counts)
    rng = np.random.default_rng(int(seed))
    canvas = canvas + float(background) + rng.normal(0.0, float(noise_std), size=canvas.shape)
    return np.clip(canvas, 0.0, None).astype(np.float64)


def load_standin_frame(
    standin_dir: Path,
    standin_id: str,
) -> tuple[np.ndarray | None, Path | None]:
    """加载替身实测帧 ``<standin_dir>/frames/<id>.npy`` (缺失返回 None)。"""
    path = Path(standin_dir) / "frames" / f"{standin_id}.npy"
    if not path.is_file():
        return None, None
    return np.asarray(np.load(path), dtype=np.float64), path


def load_standin_phase(standin_dir: Path, standin_id: str) -> np.ndarray | None:
    """加载替身相位 ``<standin_dir>/phase/<id>.npy`` (缺失返回 None)。

    dry-run 默认 ``n_pupil=128`` 与替身运行的控制网格一致 —— 形状对得上时
    用**替身相位**做预测, 保证「预测的相位」与「产生替身帧的相位」严格同一张,
    配对不掺构造差异。
    """
    path = Path(standin_dir) / "phase" / f"{standin_id}.npy"
    if not path.is_file():
        return None
    return np.asarray(np.load(path), dtype=np.float64)


def load_standin_spot_xy(standin_dir: Path) -> tuple[float, float] | None:
    """从替身运行的 ``plan.json`` 读同帧定心的 ``spot_xy`` (x=列, y=行)。"""
    plan_path = Path(standin_dir) / "plan.json"
    if not plan_path.is_file():
        return None
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    xy = plan.get("spot_xy")
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        return (float(xy[0]), float(xy[1]))
    return None


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def radial_profile(
    image: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    nbins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """绕模型网格直流格 (``M//2``) 的径向平均剖面。

    ``mask`` (布尔) 之外的格不进平均 —— 覆盖掩膜外的 0 是零填充不是测量。
    空 bin 记 NaN。Returns ``(r_centers, profile)``, r 单位 = 模型格采样。
    """
    arr = np.asarray(image, dtype=np.float64)
    m = arr.shape[0]
    c = m // 2
    nbin = int(nbins) if nbins else max(int(m // 4), 4)
    yy, xx = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
    rr = np.hypot(yy - c, xx - c)
    edges = np.linspace(0.0, float(m) / 2.0, nbin + 1)
    idx = np.clip(np.digitize(rr.ravel(), edges) - 1, 0, nbin - 1)
    valid = np.ones(arr.size, dtype=bool) if mask is None else np.asarray(mask).ravel()
    valid = valid & np.isfinite(arr.ravel())
    w = valid.astype(np.float64)
    sums = np.bincount(idx, weights=arr.ravel() * w, minlength=nbin)
    counts = np.bincount(idx, weights=w, minlength=nbin)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = np.where(counts > 0, sums / np.maximum(counts, 1e-12), np.nan)
    return 0.5 * (edges[:-1] + edges[1:]), prof


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Pearson/Spearman 相关, 常量序列或样本不足时返回 NaN。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or float(np.std(a)) <= 0 or float(np.std(b)) <= 0:
        return float("nan")
    fn = spearmanr if method == "spearman" else pearsonr
    return float(fn(a, b).statistic)


#: ``spearman_signal`` 的默认信号门限 (占覆盖区内预测峰值的比例)。
#:
#: 为什么要有门限: 远场是 ``sinc²``/Airy 型, 覆盖区里 **>99.9%** 的格强度低于
#: 峰值的千分之一 —— 那些格在 CCD 上就是背景 + 读出噪声, 它们的**秩**是随机的。
#: 不加门限的 ``spearman_coverage`` 于是被噪声底主宰: dry-run 里拿「由预测本身
#: 合成的帧」去自检, ``spearman_coverage`` 只有 0.002~0.07 (本该 ≈1), 而
#: 门限 1e-3 时自检回到 0.991~0.999 (噪声 std=6 counts / 峰值 2000)。门限再低
#: (1e-4) 自检就掉到 0.95, 再高 (1e-2) 样本数只剩几百格 —— 1e-3 是自检与样本量
#: 的折中, 可用 ``--signal-floor`` 改。
DEFAULT_SIGNAL_FLOOR = 1e-3


def pattern_metrics(
    opt_mod: Any,
    *,
    pred: np.ndarray,
    meas: np.ndarray,
    coverage: np.ndarray,
    k_mean: float,
    spec: Any,
    signal_floor: float = DEFAULT_SIGNAL_FLOOR,
) -> dict[str, Any]:
    """单图样的预测 vs 实测一致性指标 (纯数值, dry-run 与硬件共用)。

    在 opt ``compare_farfields`` (整格 Pearson、FWHM、峰值偏移、原始峰值比,
    轴向口径 8 选 1 择优) 之上补阶梯专属指标:

    * ``spearman_signal`` (**主判据**): 信号区 (覆盖掩膜 ∩ ``pred ≥
      signal_floor·pred_max``) 内逐格 Spearman。对强度定标不敏感, 又不被
      CCD 噪声底主宰 —— 怀疑的缺失离焦项会先在这里现形 (模型一个尖斑,
      硬件一个环/弥散斑 ⇒ 信号区里的秩顺序立刻乱)。见
      :data:`DEFAULT_SIGNAL_FLOOR` 的取值依据。
    * ``pearson_signal``: 同一信号区的 Pearson (仍会被最亮那几格主导, 参考用);
    * ``spearman_coverage`` / ``pearson_coverage``: 覆盖掩膜内**全部**格
      (含噪声底) —— 保留是为了与其它脚本口径可比, **不是**判据;
    * ``radial_correlation``: 径向剖面 (掩膜内) 的 Pearson。
    """
    cmp_best = sweep_axis_convention(opt_mod, pred, meas)
    aligned = apply_axis(meas, cmp_best)
    aligned_cov = apply_axis(coverage, cmp_best).astype(bool)

    pred_arr = np.asarray(pred, dtype=np.float64)
    p_in = pred_arr[aligned_cov]
    m_in = aligned[aligned_cov]
    r_pred, prof_pred = radial_profile(pred, mask=aligned_cov)
    _, prof_meas = radial_profile(aligned, mask=aligned_cov)

    # 信号区: 覆盖掩膜内, 预测强度 ≥ floor · 预测峰值 (门限来自**预测**, 不来自
    # 实测 —— 用实测当门限会让「硬件多出来的弥散光」自己把判据放宽)。
    floor = float(signal_floor)
    pred_max_cov = float(p_in.max()) if p_in.size else 0.0
    if pred_max_cov > 0:
        sig_mask = aligned_cov & (pred_arr >= floor * pred_max_cov)
    else:
        sig_mask = np.zeros(aligned_cov.shape, dtype=bool)
    p_sig = pred_arr[sig_mask]
    m_sig = aligned[sig_mask]

    model_fwhm = float(cmp_best["model_fwhm_px"])
    meas_fwhm = float(cmp_best["measured_fwhm_px"])
    dx, dy = (float(v) for v in cmp_best["peak_offset_samples"])
    scale_ccd = spec.far_field_sample_ccd_px(k_mean)
    coverage_frac = float(aligned_cov.mean()) if aligned_cov.size else float("nan")
    n_cov = int(aligned_cov.sum())
    n_sig = int(sig_mask.sum())

    row: dict[str, Any] = {
        "orientation_match": cmp_best["orientation_match"],
        "signal_floor": floor,
        "spearman_signal": _safe_corr(p_sig, m_sig, "spearman"),
        "pearson_signal": _safe_corr(p_sig, m_sig, "pearson"),
        "signal_pixels": n_sig,
        "signal_frac_of_coverage": (float(n_sig / n_cov) if n_cov else float("nan")),
        "spearman_coverage": _safe_corr(p_in, m_in, "spearman"),
        "pearson_coverage": _safe_corr(p_in, m_in, "pearson"),
        "pearson_fullgrid": float(cmp_best["intensity_correlation"]),
        "radial_correlation": _safe_corr(prof_pred, prof_meas, "pearson"),
        "model_fwhm_px": model_fwhm,
        "measured_fwhm_px": meas_fwhm,
        "fwhm_ratio": (
            float(meas_fwhm / model_fwhm) if model_fwhm > 0 else float("nan")
        ),
        "peak_offset_samples": [dx, dy],
        "peak_offset_model_px": float(math.hypot(dx, dy)),
        "peak_offset_ccd_px": float(math.hypot(dx, dy) * scale_ccd),
        "model_peak": float(cmp_best["model_peak"]),
        "measured_peak": float(cmp_best["measured_peak"]),
        "peak_ratio_raw": float(cmp_best["peak_ratio_raw"]),
        "coverage_frac": coverage_frac,
        "radial_r": r_pred.tolist(),
        "radial_pred": prof_pred.tolist(),
        "radial_meas": prof_meas.tolist(),
    }
    return row


# ---------------------------------------------------------------------------
# 图 (fidlad_* 前缀)
# ---------------------------------------------------------------------------
def plot_radial_profiles(
    entries: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    """逐图样径向剖面小图阵: ``fidlad_radial.png``。

    ``entries`` 每项 ``(pattern_id, r, pred_prof, meas_prof)``; 两侧剖面各按
    自身峰值归一 (模型无量纲 vs CCD counts, 不同单位不能直接叠)。
    """
    count = len(entries)
    ncols = min(3, max(count, 1))
    nrows = int(math.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax, (pid, r, prof_pred, prof_meas) in zip(axes.ravel(), entries, strict=False):
        pp = np.asarray(prof_pred, dtype=np.float64)
        pm = np.asarray(prof_meas, dtype=np.float64)
        pp_n = pp / max(float(np.nanmax(pp)), 1e-12) if np.isfinite(pp).any() else pp
        pm_n = pm / max(float(np.nanmax(pm)), 1e-12) if np.isfinite(pm).any() else pm
        ax.plot(r, pp_n, "-", color="#2f6fb0", label="模型预测")
        ax.plot(r, pm_n, "--", color="#c0504d", label="实测 (重采样)")
        ax.set_title(pid, fontsize=10)
        ax.set_xlabel("半径 (模型格采样)")
        ax.set_ylabel("I / I_max")
        ax.legend(fontsize=8)
    for ax in axes.ravel()[count:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_correlation_bars(rows: list[dict[str, Any]], path: Path) -> None:
    """逐图样相关系数横条图: ``fidlad_corr.png``。"""
    ids = [str(r["pattern_id"]) for r in rows]
    keys = (
        ("spearman_signal", "Spearman (信号区, 主判据)", "#2f6fb0"),
        ("spearman_coverage", "Spearman (覆盖掩膜内, 含噪声底)", "#9db9d8"),
        ("radial_correlation", "径向剖面 Pearson", "#c0504d"),
    )
    y = np.arange(len(ids), dtype=np.float64)
    height = 0.26
    fig, ax = plt.subplots(figsize=(9.0, 1.1 + 0.55 * len(ids)))
    for k, (key, label, color) in enumerate(keys):
        vals = [
            float(r[key]) if r.get(key) is not None and np.isfinite(float(r[key])) else 0.0
            for r in rows
        ]
        ax.barh(y + (k - 1) * height, vals, height=height, label=label, color=color)
    ax.set_yticks(y, ids)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.8, color="green", linewidth=0.8, linestyle=":")
    ax.set_xlabel("相关系数 (绿虚线 = 0.8 参考线)")
    ax.set_title("预测 vs 实测: 逐图样相关系数")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 落盘
# ---------------------------------------------------------------------------
def phase_stats(phase: np.ndarray) -> dict[str, float]:
    """相位块的 RMS / PTV (弧度)。"""
    arr = np.asarray(phase, dtype=np.float64)
    return {
        "rms_rad": float(arr.std()),
        "ptv_rad": float(arr.max() - arr.min()),
        "shape_rows": int(arr.shape[0]),
        "shape_cols": int(arr.shape[1]),
    }


def write_metrics(out_dir: Path, opt_mod: Any, rows: list[dict[str, Any]]) -> None:
    """``metrics.json`` (全字段) + ``metrics.csv`` (扁平列, 复用 opt write_csv)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    fieldnames = [
        "pattern_id",
        "source",
        "measured_source",
        "note",
        "orientation_match",
        "signal_floor",
        "spearman_signal",
        "pearson_signal",
        "signal_pixels",
        "signal_frac_of_coverage",
        "spearman_coverage",
        "pearson_coverage",
        "pearson_fullgrid",
        "radial_correlation",
        "model_fwhm_px",
        "measured_fwhm_px",
        "fwhm_ratio",
        "peak_offset_model_px",
        "peak_offset_ccd_px",
        "model_peak",
        "measured_peak",
        "peak_ratio_raw",
        "coverage_frac",
        "exposure_ms",
    ]
    flat = [{k: r.get(k) for k in fieldnames} for r in rows]
    opt_mod.write_csv(out_dir / "metrics.csv", fieldnames, flat)


def write_summary_md(
    out_dir: Path,
    *,
    mode: str,
    plan: dict[str, Any],
    rows: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    occupancy: dict[str, Any] | None = None,
) -> Path:
    """``ladder_summary.md``: 阶梯总表 + 设备纪律块 + 解读指引。"""

    def fmt(v: Any, spec: str = ".4g") -> str:
        if v is None:
            return "—"
        if isinstance(v, float) and not math.isfinite(v):
            return "nan"
        if isinstance(v, float):
            return format(v, spec)
        return str(v)

    lines = [
        f"# 物理预测模型保真度阶梯 ({mode})",
        "",
        f"- 时间戳 (UTC): {plan.get('timestamp_utc')}",
        f"- 模式: {mode}"
        + (" (离线演练: 替身/合成帧, 未打开任何设备)" if mode == "dry-run" else " (硬件实测)"),
        f"- 几何标定: {plan.get('geometry', {}).get('path')}"
        f" (k_mean={fmt(plan.get('geometry', {}).get('k_mean'))} px/panel-px,"
        f" rotation={fmt(plan.get('geometry', {}).get('rotation_deg'), '.3f')}°)",
        f"- 模型网格: N={plan.get('model', {}).get('n_pupil')},"
        f" M={plan.get('model', {}).get('m_grid')},"
        f" dp={fmt(plan.get('model', {}).get('dp_panel_px'))} panel-px",
        f"- 重采样中心 (x, y): {fmt(plan.get('centre_xy', [None, None])[0], '.1f')},"
        f" {fmt(plan.get('centre_xy', [None, None])[1], '.1f')}",
        f"- 预测来源: {plan.get('predict_from')}",
        f"- 信号门限 (spearman_signal 用): {fmt(plan.get('signal_floor'), '.3g')}"
        " × 覆盖区内预测峰值",
        "",
        "## 阶梯结果",
        "",
        "| 图样 | 实测来源 | Spearman(信号) | Pearson(信号) | 信号格数 | Spearman(覆盖) | 径向 r | FWHM 比 | 峰偏 (格/CCD px) | 覆盖 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {pid} | {msrc} | {sps} | {pes} | {nsig} | {spc} | {rc} | {fr} | {po} | {cov} |".format(
                pid=r.get("pattern_id"),
                msrc=r.get("measured_source"),
                sps=fmt(r.get("spearman_signal")),
                pes=fmt(r.get("pearson_signal")),
                nsig=r.get("signal_pixels"),
                spc=fmt(r.get("spearman_coverage")),
                rc=fmt(r.get("radial_correlation")),
                fr=fmt(r.get("fwhm_ratio")),
                po=f"{fmt(r.get('peak_offset_model_px'), '.1f')}"
                f" / {fmt(r.get('peak_offset_ccd_px'), '.1f')}",
                cov=fmt(r.get("coverage_frac"), ".2%"),
            )
        )
    if skipped:
        lines += ["", "## 跳过的图样", ""]
        lines += [f"- {s['pattern_id']}: {s['reason']}" for s in skipped]
    if occupancy is not None:
        lines += [
            "",
            "## 设备纪律 (hardware_discipline)",
            "",
            f"- 占用设备: {occupancy.get('devices')}",
            f"- 占用开始 (UTC): {occupancy.get('start_utc')}",
            f"- 占用结束 (UTC): {occupancy.get('end_utc')}",
            f"- 是否正常关闭: {occupancy.get('closed_normally')}",
            f"- SLM 开/关次数: {occupancy.get('slm_opens')}/{occupancy.get('slm_closes')}",
            f"- 相机开/关次数: {occupancy.get('cam_opens')}/{occupancy.get('cam_closes')}",
        ]
    lines += [
        "",
        "## 解读指引",
        "",
        "- **主判据是 `Spearman(信号)`** —— 覆盖区里预测强度 ≥ 门限×预测峰值",
        "  的那些格的秩相关。远场是 sinc²/Airy 型, 覆盖区里 **>99.9%** 的格在",
        "  CCD 上只是背景+读出噪声, 它们的**秩**是随机的; 不设门限的",
        "  `Spearman(覆盖)` 于是被噪声底主宰 —— dry-run 里拿「由预测本身合成的",
        "  帧」自检, `Spearman(覆盖)` 只有 0.002~0.07 (本该 ≈1), 门限 1e-3 时",
        "  `Spearman(信号)` 回到 0.991~0.999 (噪声 std=6 counts / 峰值 2000",
        "  counts)。所以 `Spearman(覆盖)` 只作参考列, 不要拿它下结论。",
        "- **缺失离焦项的指纹**: 若 `defocus_z4_pos` / `defocus_z4_neg` 的",
        "  `Spearman(信号)` 显著低于 `flat`, 且 FWHM 比偏离 1 (实测更宽/更窄)、",
        "  峰值偏移随离焦符号反号 —— 说明模型传播距离与实验系统焦面失配, 即",
        "  物理预测模型缺一个离焦 (二次相位) 自由度; `tilt_x/y` 的峰值偏移给出轴",
        "  标定残差作对照。",
        "- **信号门限来自预测而不是实测**: 用实测当门限会让「硬件多出来的弥散",
        "  光」自己把判据放宽, 假设就检不出来了。门限值写在表头与 `plan.json`。",
        "- 覆盖掩膜外的格是重采样**零填充**, 未进任何相关系数; `覆盖` 列给出",
        "  掩膜占比, `信号格数` 给出门限内的格数 (太少则 Spearman 不稳)。",
        "- 轴向口径 (`orientation_match`) 是 8 种转置/镜像里按整格 Pearson",
        "  择优的结果 —— 与 hw_differentiable_phase_opt 同一套约定。",
        "",
    ]
    path = Path(out_dir) / "ladder_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def copy_sources(out_dir: Path, files: list[Path]) -> None:
    """把脚本源码自拷贝进产物目录 (溯源)。"""
    dst = Path(out_dir) / "source"
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        f = Path(f)
        if f.is_file():
            shutil.copy2(f, dst / f.name)
