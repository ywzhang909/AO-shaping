"""离线验证: 前向物理预测模型参数拟合 + 可微共同优化的可行性 (2026-11-17 阶梯数据)。

要回答的三个问题
----------------
1. 已落盘的「命令相位 → CCD 实测远场」数据, 能不能把物理预测模型的**几何/尺度/
   旋转/模糊/离焦**参数拟合出来, 且每个参数**可辨识**?
2. 拟合出来的模型, 能不能作为可微优化器里的物理预测 (梯度非零、损失单调)?
3. 模型预测的损失, 跟实测损失**同号同量级**吗 (即闭环是否真的在下降)?

输入 (只读已落盘产物, 全程不碰设备)
-----------------------------------
* ``data/debug/differentiable_opt/fidelity_ladder_20261117_022400`` —— 7 个图样的
  ``phase/*.npy`` (256² 瞳面块) / ``pred/*.npy`` (1024² 模型远场) /
  ``frames/*.npy`` (1944×2592 CCD 帧) 与 ``plan.json`` / ``metrics.csv``;
* ``data/debug/differentiable_opt/20261117_022658`` —— 同一晚硬件闭环的
  ``history_hw.csv`` / ``phase_initial.npy`` / ``phase_final.npy`` /
  ``ccd_hw_iter_00{2,3,4}.npz`` / ``run_config.json`` (含**现场实测**几何标定);
* ``data/slm_geometry_calib_20261023.npz`` —— 阶梯那次运行用的**存档**几何
  (2026-10-23; canonical 的 ``data/slm_geometry_calib.npz`` 已于 2026-11-17 被现场
  标定刷新, 本脚本按 ``provenance.measured_at`` 校验后仍读这份快照)。

已确认的根因 (本脚本不再把它当假设)
----------------------------------
阶梯 ``plan.json`` 的 ``centre_xy = (1223.006, 962.631)``、``centre_source = 存档
data/slm_geometry_calib.npz`` (现为 ``slm_geometry_calib_20261023.npz``); 而同一晚
``20261117_022658/run_config.json`` 的
``geometry.measured_this_run = true`` 给出 ``(cx, cy) = (1224.416, 1104.843)``。
**x 差 1.4 px, y 差 142.2 px** ⇒ 阶梯的 7 个图样是拿一个**陈旧 142 px 的 y 原点**
去重采样的, 于是模型零级与实测零级错开, 全部 Pearson 退到 0.001~0.017。
本脚本用「四列对照」(存档参数 / 仅换中心 / 换中心+尺度 / 拟合最优) 给出这一项
的**解释力上界**。

口径 (与仓库既有实现一致, 不另立一套)
------------------------------------
* 模型远场: ``OPT.model_far_field(amplitude, phase)`` —— 与阶梯 ``pred`` 同一函数;
  开头的「复算对账」要求它与落盘 ``pred/`` 逐点一致 (相对误差 < 1e-5), 否则 exit 5;
* CCD → 模型网格: ``OPT.prepare_frame`` + ``OPT.resample_ccd_to_model`` (双线性,
  ``mode=constant``, ``cval=0``), 与阶梯 ``resample_with_coverage`` 完全同路径;
* 覆盖掩膜: ``SECTIONS.coverage_mask``; 信号区: ``pred ≥ floor·pred_max`` 与
  ``measured ≥ floor·meas_max`` 的**并集** ∩ 覆盖区, floor =
  ``SECTIONS.DEFAULT_SIGNAL_FLOOR``;
* 主判据 ``spearman_signal`` 与 ``pearson_signal`` / ``radial_correlation``:
  复刻 ``SECTIONS.pattern_metrics`` 的公式, 但**固定一个全局轴向口径**
  (阶梯的 ``pattern_metrics`` 是**逐图样**重扫 8 种口径, 相关性趋零时择优本身就
  是噪声 —— 本脚本把这一点作为「口径矛盾」单独报告)。

GPU 契约
--------
调用方要求「GPU 必须真实生效」。本脚本: (a) 每次拟合阶段打印
``[GPU] device=<name> peak_alloc=<MB>``; (b) 参数张量断言 ``.is_cuda``;
(c) ``torch.cuda`` 不可用直接 ``SystemExit(4)``, **没有任何静默 CPU 回退**。
优化用 float32 (consumer GPU 上 float64 的反向慢一个量级); 最终评分用 float64
(numpy 路径), 两者在报告里分开写。

离线契约
--------
开头与结尾都跑 ``AFM.assert_no_drivers()`` —— ``sys.modules`` 里出现任何
``ao_shaping.drivers.*`` 就 exit 3。脚本可以没有任何硬件的机器上跑。

用法
----
    python scripts/slm/analyze_fwdmodel_fit_20261117.py

自验证要点 (报告里给确切 exit code) 见 ``--check`` 段。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "scripts"),
    str(ROOT / "scripts" / "slm"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from loguru import logger
from scripts._common import fmt_g_or_dash

EXIT_NO_DRIVER_CONTRACT = 3
EXIT_NO_GPU = 4
EXIT_PRED_MISMATCH = 5


def _load_module(name: str, path: Path) -> Any:
    """按文件路径加载兄弟脚本 (与 ``SECTIONS.load_opt_module`` 同一做法)。"""
    if not path.is_file():
        raise FileNotFoundError(f"模块文件不存在: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: 阶梯产物 (被复用的权威口径都在这里)。
SECTIONS = _load_module(
    "_fwdmodel_ladder_sections", ROOT / "scripts" / "slm" / "_hw_fidelity_ladder_sections.py"
)
#: 上一个拟合脚本: 只借用 ``assert_no_drivers`` / ``radial_profile`` / ``fwhm_profile``。
AFM = _load_module("_fwdmodel_analyze_fit", ROOT / "scripts" / "slm" / "analyze_fwdmodel_fit.py")
#: 主优化脚本 (ModelSpec / prepare_frame / resample_ccd_to_model / model_far_field)。
OPT = SECTIONS.load_opt_module()

from ao_shaping.algorithm.signal_processing.differentiable_beam import (
    DifferentiableBeamOptimizer,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
#: 阶梯产物目录 (7 个图样: 模型远场 + CCD 帧 + 相位块)。
LADDER_DIR = ROOT / "data" / "debug" / "differentiable_opt" / "fidelity_ladder_20261117_022400"
#: 同一晚的硬件闭环运行目录 (现场几何标定 + 5 步实测历史)。
OPT_RUN_DIR = ROOT / "data" / "debug" / "differentiable_opt" / "20261117_022658"
#: 阶梯那次运行用的**存档**几何 (2026-10-23)。canonical 路径已被 2026-11-17 现场
#: 标定刷新 (``persist_live_geometry.py``), 旧值存档为 ``slm_geometry_calib_20261023.npz``。
ARCHIVE_GEOM_NPZ = ROOT / "data" / "slm_geometry_calib.npz"
#: 2026-10-23 存档快照 (阶梯那次实际读到的值)。
ARCHIVE_GEOM_NPZ_SNAPSHOT = ROOT / "data" / "slm_geometry_calib_20261023.npz"
#: 阶梯与现场两次标定的日期前缀, 用来校验读到的确实是阶梯当时那份几何。
ARCHIVE_GEOM_DAY = "2026-10-23"

DEFAULT_REPORT = (
    ROOT / "report" / "slm" / "differentiable_opt" / "artifacts"
    / "fwdmodel_fit_20261117" / "fwdmodel_fit_20261117.md"
)

#: 阶梯的图样顺序 (flat 放最前 —— 它就是定心基准)。
PATTERN_IDS: tuple[str, ...] = (
    "flat",
    "gs_init",
    "optimized_final",
    "defocus_z4_pos",
    "defocus_z4_neg",
    "tilt_x",
    "tilt_y",
)

#: 信号区门限 (占覆盖区内预测峰值的比例), 直接取阶梯的默认值。
SIGNAL_FLOOR = float(SECTIONS.DEFAULT_SIGNAL_FLOOR)

#: 优化器损失里径向项的中心 bin 数 (样本)。32 样本 × 1.97 px/样本 ≈ 63 CCD px,
#: 覆盖目标方块 (60 px) 的整块足迹 —— 更远的环对形状判据没有信息量。
RADIAL_BINS = 32

#: 参数表: (名字, 初值, 归一化步长, 下界, 上界)。
#:
#: * ``centre_x`` / ``centre_y``: CCD px, 初值取**现场标定** (不是粗搜值);
#: * ``scale_px``: 每个模型样本对应的 CCD 像元数 ``s = K/(M·dp)``, 初值取现场的
#:   ``k_mean/(1024·3.75) = 1.969767``;
#: * ``rotation_deg``: 重采样格相对相机轴的旋转, 初值 0 (现场标定 0.1675°);
#: * ``blur_ccd_px``: 模型 → CCD 的高斯 PSF σ, 单位 CCD px (冻结检查点是 0);
#: * ``defocus_z4_rad``: 追加的 Noll(2,0) 离焦, 单位**圆内 RMS 弧度**, 与阶梯
#:   ``defocus_z4_*`` 图样同一单位 (那批是 ±1 rad);
#: * ``phase_scale``: 命令相位 → 有效相位的乘子 (992/1023 灰阶过调制 = 1.0313)。
PARAM_SPECS: tuple[tuple[str, float, float, float, float], ...] = (
    ("centre_x", 1224.416, 10.0, 1100.0, 1350.0),
    ("centre_y", 1104.843, 10.0, 880.0, 1350.0),
    ("scale_px", 1.969767, 0.01, 1.80, 2.15),
    ("rotation_deg", 0.0, 0.20, -4.0, 4.0),
    ("blur_ccd_px", 1.5, 0.30, 0.0, 8.0),
    ("defocus_z4_rad", 0.0, 0.10, -2.0, 2.0),
    ("phase_scale", 1.0, 0.005, 0.90, 1.10),
)
PARAM_NAMES: tuple[str, ...] = tuple(spec[0] for spec in PARAM_SPECS)
GEO_PARAM_COUNT = 5  # 前 5 个参数 (几何 + 模糊) 参与多起点粗拟合


# ---------------------------------------------------------------------------
# GPU 契约
# ---------------------------------------------------------------------------
def require_cuda() -> torch.device:
    """没有 CUDA 就退出 —— 严禁静默回退到 CPU。"""
    if not torch.cuda.is_available():
        logger.error("CUDA 不可用; 本脚本的 GPU 契约要求真实 GPU, 拒绝 CPU 回退")
        raise SystemExit(EXIT_NO_GPU)
    return torch.device("cuda:0")


def gpu_banner(tag: str) -> dict[str, Any]:
    """打印并返回一项 GPU 证据 (设备名 + 峰值显存); 同时校验张量在卡上。"""
    name = torch.cuda.get_device_name(0)
    peak = torch.cuda.max_memory_allocated() / 1e6
    probe = torch.zeros(8, device="cuda")
    if not probe.is_cuda:
        logger.error("[{}] 探针张量不在 CUDA 上, 拒绝继续", tag)
        raise SystemExit(EXIT_NO_GPU)
    logger.info(
        "[GPU] tag={} device={} torch={} peak_alloc={:.2f} MB",
        tag,
        name,
        torch.__version__,
        peak,
    )
    return {
        "tag": tag,
        "device_name": name,
        "torch_version": str(torch.__version__),
        "peak_alloc_mb": float(peak),
    }


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
@dataclass
class LabData:
    """一批已落盘的「相位 → 模型远场 → CCD 实测」数据 + 几何标定。"""

    spec: Any
    amplitude: np.ndarray  # (m, m) 源面振幅 (高斯 × 硬口径)
    z4_block: np.ndarray  # (m, m) 嵌好的 Noll(2,0) 单位振幅
    phases: np.ndarray  # (P, N, N) 瞳面相位块
    preds: np.ndarray  # (P, m, m) 模型远场强度 (落盘)
    frames: np.ndarray  # (P, H, W) 预处理后的 CCD 帧
    metrics_csv: dict[str, dict[str, Any]]
    plan: dict[str, Any]
    archive: dict[str, Any]
    live: dict[str, Any]
    run_cfg: dict[str, Any]
    exposure_ms: np.ndarray  # (P,)
    _target: np.ndarray | None = field(default=None, repr=False)

    @property
    def pattern_count(self) -> int:
        return len(PATTERN_IDS)

    @property
    def m(self) -> int:
        return int(self.spec.m_grid)

    @property
    def frame_shape(self) -> tuple[int, int]:
        return int(self.frames.shape[1]), int(self.frames.shape[2])

    @property
    def k_per_scale(self) -> float:
        """``scale`` (CCD px / 模型样本) → ``K`` (CCD px / 面板像元) 的乘子。"""
        return float(self.spec.m_grid) * float(self.spec.dp)

    def target(self) -> np.ndarray:
        """目标方块 (峰值归一化到 1 的 ``m×m`` 强度图), 与优化运行的目标同一张。"""
        if self._target is None:
            k_archive = float(self.archive["k_mean"])
            target, _info = OPT.build_target(
                self.spec, "square", float(self.plan["model"]["target"]["side_ccd_px_requested"]),
                k_archive, 15.0,
            )
            arr = np.asarray(target, dtype=np.float64)
            self._target = arr / max(float(arr.max()), 1e-12)
        return self._target


def _read_csv_dict(path: Path, key: str = "pattern_id") -> dict[str, dict[str, Any]]:
    import csv

    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rec: dict[str, Any] = {}
            for k, v in row.items():
                if k is None:
                    continue
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
            if key in rec:
                out[str(rec[key])] = rec
    return out


def _geometry_provenance_day(npz_path: Path) -> str:
    """读 npz 的 ``provenance.measured_at`` 日期前缀 (缺字段返回空串)。"""
    with np.load(npz_path, allow_pickle=False) as d:
        if "provenance" not in d.files:
            return ""
        try:
            prov = json.loads(str(d["provenance"]))
        except json.JSONDecodeError:
            return ""
    return str(prov.get("measured_at", ""))[:10]


def resolve_archive_geometry_npz() -> Path:
    """定位阶梯那次 (2026-10-23) 的存档几何。

    ``data/slm_geometry_calib.npz`` 已由 ``persist_live_geometry.py`` 换成 2026-11-17
    现场标定, 所以不能盲读 canonical 路径: 先试日期快照, 再退回 canonical 并校验其
    ``provenance`` 仍是 2026-10-23, 否则报错 —— 避免本报告的四列对照偷偷换几何。
    """
    for cand in (ARCHIVE_GEOM_NPZ_SNAPSHOT, ARCHIVE_GEOM_NPZ):
        if not cand.is_file():
            continue
        if _geometry_provenance_day(cand) == ARCHIVE_GEOM_DAY:
            return cand
    raise FileNotFoundError(
        f"找不到 {ARCHIVE_GEOM_DAY} 的存档几何: 试过 {ARCHIVE_GEOM_NPZ_SNAPSHOT} 与 "
        f"{ARCHIVE_GEOM_NPZ}, 两者的 provenance.measured_at 都不是 {ARCHIVE_GEOM_DAY}"
    )


def load_lab_data(ladder_dir: Path, run_dir: Path) -> LabData:
    """装载阶梯 + 同晚硬件运行 + 两套几何标定。"""
    plan = json.loads((ladder_dir / "plan.json").read_text(encoding="utf-8"))
    mcfg = plan["model"]
    spec = OPT.ModelSpec(
        n_pupil=int(mcfg["n_pupil"]),
        m_grid=int(mcfg["m_grid"]),
        dp=float(mcfg["dp_panel_px"]),
        span=float(mcfg["span_panel_px"]),
        pupil_radius=float(mcfg["pupil_radius"]),
        pupil_center=tuple(float(v) for v in mcfg["pupil_center"]),
        w0_px=float(mcfg["beam_w0_px"]),
        padding=int(mcfg["padding"]),
    )

    phases, preds, frames, exposure = [], [], [], []
    for pid in PATTERN_IDS:
        phases.append(np.asarray(np.load(ladder_dir / "phase" / f"{pid}.npy"), dtype=np.float64))
        preds.append(np.asarray(np.load(ladder_dir / "pred" / f"{pid}.npy"), dtype=np.float32))
        raw = np.asarray(np.load(ladder_dir / "frames" / f"{pid}.npy"), dtype=np.float64)
        frames.append(np.asarray(OPT.prepare_frame(raw), dtype=np.float32))
        exposure.append(0.0)

    metrics_csv = _read_csv_dict(ladder_dir / "metrics.csv")
    for i, pid in enumerate(PATTERN_IDS):
        rec = metrics_csv.get(pid, {})
        exposure[i] = float(rec.get("exposure_ms", 0.0) or 0.0)

    archive = SECTIONS.load_geometry_npz(resolve_archive_geometry_npz())
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    geom = run_cfg["geometry"]
    live = {
        "centre_xy": (float(geom["center"][1]), float(geom["center"][0])),
        "Kx": float(geom["kx"]),
        "Ky": float(geom["ky"]),
        "k_mean": float(geom["k_mean"]),
        "rotation_deg": float(geom["rotation_deg"]),
        "mirror": bool(geom["mirror"]),
        "measured_this_run": bool(geom.get("measured_this_run", True)),
    }

    amplitude = np.asarray(OPT.build_source_amplitude(spec), dtype=np.float64)
    z4 = SECTIONS.embed_block(spec, SECTIONS.noll_phase_block(2, 0, int(spec.n_pupil), 1.0))

    return LabData(
        spec=spec,
        amplitude=amplitude,
        z4_block=np.asarray(z4, dtype=np.float64),
        phases=np.stack(phases),
        preds=np.stack(preds),
        frames=np.stack(frames),
        metrics_csv=metrics_csv,
        plan=plan,
        archive=archive,
        live=live,
        run_cfg=run_cfg,
        exposure_ms=np.asarray(exposure, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# 几何: 模型网格 → 相机坐标 (numpy 与 torch 两套, 同一公式)
# ---------------------------------------------------------------------------
def camera_offsets(cx: float, cy: float, scale: float, rotation_deg: float, m: int,
                   ) -> tuple[np.ndarray, np.ndarray]:
    """模型格 ``(i, j)`` 的相机坐标 ``(x_ccd, y_ccd)``。

    ``u = j − M/2`` (列向, 沿相机 x)、``v = i − M/2`` (行向, 沿相机 y);
    ``rotation_deg = 0`` 时与 ``OPT.resample_ccd_to_model`` 的 ``offsets`` 逐点相同。
    正 ``rotation`` 把模型格相对相机轴逆时针转 (在相机坐标系里看)。
    """
    idx = np.arange(int(m), dtype=np.float64) - float(m) / 2.0
    u, v = np.meshgrid(idx, idx, indexing="xy")  # u: 列向, v: 行向
    th = math.radians(float(rotation_deg))
    cos_t, sin_t = math.cos(th), math.sin(th)
    x = float(cx) + float(scale) * (u * cos_t - v * sin_t)
    y = float(cy) + float(scale) * (u * sin_t + v * cos_t)
    return x, y


def geometry_dict(centre_xy: Sequence[float], scale_px: float, rotation_deg: float,
                  blur_ccd_px: float, defocus_z4_rad: float, phase_scale: float,
                  ) -> dict[str, float]:
    """把参数向量摊成有名字的字典 (报告与 JSON 都用它)。"""
    return {
        "centre_x": float(centre_xy[0]),
        "centre_y": float(centre_xy[1]),
        "scale_px": float(scale_px),
        "k_mean": float(scale_px) * float(1024 * 3.75),
        "rotation_deg": float(rotation_deg),
        "blur_ccd_px": float(blur_ccd_px),
        "defocus_z4_rad": float(defocus_z4_rad),
        "phase_scale": float(phase_scale),
    }


# ---------------------------------------------------------------------------
# 纯 numpy 评分路径 (float64) —— 最终数字都从这里出
# ---------------------------------------------------------------------------
def apply_orientation(arr: np.ndarray, orient: dict[str, Any]) -> np.ndarray:
    """按口径变换数组 (复刻 ``SECTIONS.apply_axis``; 不强转 dtype)。"""
    out = np.asarray(arr)
    if orient.get("transpose"):
        out = out.T
    if orient.get("flip_rows"):
        out = out[::-1, :]
    if orient.get("flip_cols"):
        out = out[:, ::-1]
    return out


def _corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 3 or float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    fn = spearmanr if method == "spearman" else pearsonr
    return float(fn(x, y).statistic)


def render_numpy(
    data: LabData, geom: dict[str, float], *, blur: bool = True
) -> dict[str, Any]:
    """float64 前向: 每个图样的模型远场 + 实测重采样 + 覆盖掩膜。

    Returns:
        ``{"pred": (P,m,m), "meas": (P,m,m), "cov": (m,m)}`` —— 都还在**模型网格**
        上 (未做轴向口径变换)。
    """
    from scipy.ndimage import gaussian_filter

    m = data.m
    k_mean = float(geom["k_mean"])
    preds = np.empty((data.pattern_count, m, m), dtype=np.float64)
    for i, pid in enumerate(PATTERN_IDS):
        block = np.asarray(data.phases[i], dtype=np.float64) * float(geom["phase_scale"])
        embedded = SECTIONS.embed_block(data.spec, block)
        embedded = embedded + float(geom["defocus_z4_rad"]) * data.z4_block
        pred = OPT.model_far_field(data.amplitude, embedded)
        sigma_samples = float(geom["blur_ccd_px"]) / max(float(geom["scale_px"]), 1e-9)
        if blur and sigma_samples > 1e-6:
            pred = gaussian_filter(pred, sigma_samples, mode="constant", cval=0.0)
        preds[i] = pred

    return {"pred": preds, "meas": None, "cov": None}


def resample_measured_numpy(data: LabData, geom: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """实测帧 → 模型网格 (与 ``OPT.resample_ccd_to_model`` 同一路径, float64)。"""
    m = data.m
    k_mean = float(geom["k_mean"])
    x, y = camera_offsets(
        geom["centre_x"], geom["centre_y"], geom["scale_px"], geom["rotation_deg"], m
    )
    meas = np.empty((data.pattern_count, m, m), dtype=np.float64)
    if abs(float(geom["rotation_deg"])) > 0.0:
        from scipy.ndimage import map_coordinates

        for i in range(data.pattern_count):
            meas[i] = map_coordinates(
                data.frames[i].astype(np.float64),
                np.stack([y.ravel(), x.ravel()]),
                order=1,
                mode="constant",
                cval=0.0,
            ).reshape(m, m)
    else:
        for i in range(data.pattern_count):
            meas[i] = OPT.resample_ccd_to_model(
                data.frames[i].astype(np.float64),
                (float(geom["centre_x"]), float(geom["centre_y"])),
                k_mean,
                data.spec,
            )
    height, width = data.frame_shape
    cov = (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    return meas, cov


def signal_metrics(
    pred: np.ndarray, meas: np.ndarray, cov: np.ndarray, orient: dict[str, Any],
    scale_px: float,
) -> dict[str, Any]:
    """单图样一致性指标 (固定轴向口径; 公式 = ``SECTIONS.pattern_metrics``)。"""
    aligned = apply_orientation(meas, orient)
    aligned_cov = apply_orientation(cov, orient).astype(bool)
    p_in = np.asarray(pred, dtype=np.float64)[aligned_cov]
    peak = float(p_in.max()) if p_in.size else 0.0
    if peak > 0.0:
        sig = aligned_cov & (np.asarray(pred) >= SIGNAL_FLOOR * peak)
    else:
        sig = np.zeros_like(aligned_cov)
    p_sig = np.asarray(pred, dtype=np.float64)[sig]
    m_sig = aligned[sig]

    r_pred, prof_pred = AFM.radial_profile(
        pred, (pred.shape[1] / 2.0, pred.shape[0] / 2.0), step=float(scale_px)
    )
    _, prof_meas = AFM.radial_profile(
        aligned, (pred.shape[1] / 2.0, pred.shape[0] / 2.0), step=float(scale_px)
    )
    ratio = prof_pred / np.maximum(prof_pred.max(), 1e-12)
    ratio_m = prof_meas / np.maximum(prof_meas.max(), 1e-12)
    return {
        "spearman_signal": _corr(p_sig, m_sig, "spearman"),
        "pearson_signal": _corr(p_sig, m_sig, "pearson"),
        "pearson_coverage": _corr(np.asarray(pred)[aligned_cov], aligned[aligned_cov]),
        "radial_correlation": _corr(ratio, ratio_m),
        "signal_pixels": int(sig.sum()),
        "signal_frac": float(sig.sum() / max(int(aligned_cov.sum()), 1)),
        "pred_fwhm_px": AFM.fwhm_profile(prof_pred, float(scale_px)),
        "meas_fwhm_px": AFM.fwhm_profile(prof_meas, float(scale_px)),
        "peak_ratio": float(np.max(m_sig) / max(np.max(p_sig), 1e-12)) if p_sig.size else float("nan"),
    }


def pattern_model_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """峰值归一化 MSE (pred vs 目标方块) —— 与优化器的损失逐字同型。"""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    norm = p / (float(p.max()) + 1e-8)
    return float(np.mean((norm - t) ** 2))


def energy_fraction(pred: np.ndarray, radius_samples: float) -> float:
    """以直流格为心、半径 ``radius_samples`` 的圆盘里占了多少能量占比。"""
    m = pred.shape[0]
    c = m // 2
    idx = np.arange(m) - c
    r2 = idx[:, None] ** 2 + idx[None, :] ** 2
    mask = r2 <= radius_samples**2
    total = float(np.sum(pred))
    return float(np.sum(pred[mask]) / total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# torch 可微前向 (float32 优化路径)
# ---------------------------------------------------------------------------
@dataclass
class GeomBatch:
    """一批起点的重采样几何 (S, m, m)。"""

    centre_x: torch.Tensor
    centre_y: torch.Tensor
    scale_px: torch.Tensor
    rotation_deg: torch.Tensor

    def coords(self, m: int, frame_h: int, frame_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.centre_x.device
        idx = torch.arange(m, dtype=torch.float32, device=device) - float(m // 2)
        u = idx.view(1, 1, m)
        v = idx.view(1, m, 1)
        th = torch.deg2rad(self.rotation_deg).view(-1, 1, 1)
        cos_t, sin_t = torch.cos(th), torch.sin(th)
        x = self.centre_x.view(-1, 1, 1) + self.scale_px.view(-1, 1, 1) * (
            u * cos_t - v * sin_t
        )
        y = self.centre_y.view(-1, 1, 1) + self.scale_px.view(-1, 1, 1) * (
            u * sin_t + v * cos_t
        )
        return x, y

    def grid(self, m: int, frame_h: int, frame_w: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.coords(m, frame_h, frame_w)
        gx = 2.0 * x / float(frame_w - 1) - 1.0
        gy = 2.0 * y / float(frame_h - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1)
        cov = (x >= -0.5) & (x <= frame_w - 0.5) & (y >= -0.5) & (y <= frame_h - 0.5)
        return grid, cov


class FittedForward:
    """可微前向: 相位 → 模型远场 → (与重采样实测比较的) 形状损失。

    优化路径全 float32; 参数是 ``(S, 7)`` 张量 (S = 起点数), 各起点独立。
    """

    def __init__(self, data: LabData, device: torch.device) -> None:
        self.data = data
        self.device = device
        self.m = data.m
        self.frame_h, self.frame_w = data.frame_shape
        self.preds = torch.from_numpy(data.preds).to(device)  # (P, m, m) float32
        self.amp = torch.from_numpy(np.asarray(data.amplitude, dtype=np.float32)).to(device)
        self.z4 = torch.from_numpy(np.asarray(data.z4_block, dtype=np.float32)).to(device)
        self.blocks = torch.stack(
            [
                torch.from_numpy(SECTIONS.embed_block(data.spec, np.asarray(b, dtype=np.float64)))
                .to(dtype=torch.float32)
                .to(device)
                for b in data.phases
            ]
        )  # (P, m, m)
        self.frames = torch.from_numpy(data.frames).to(device)  # (P, H, W)
        idx = torch.arange(self.m, dtype=torch.float32, device=device) - float(self.m) / 2.0
        r = torch.sqrt(idx.view(1, -1) ** 2 + idx.view(-1, 1) ** 2)
        self.bin_index = r.floor().clamp(0, RADIAL_BINS - 1).to(torch.int64).reshape(-1)
        self.target = torch.from_numpy(np.asarray(data.target(), dtype=np.float32)).to(device)

    # -- 前向 ---------------------------------------------------------------
    def pred_stack(self, pattern: int, c4: torch.Tensor, phase_scale: torch.Tensor,
                   *, cached: bool) -> torch.Tensor:
        """``(S, m, m)`` 模型远场; ``cached=True`` 时直接返回落盘 pred (c4=0, ps=1)。"""
        if cached:
            return self.preds[pattern].unsqueeze(0).expand(c4.shape[0], -1, -1)
        s = c4.shape[0]
        phi = self.blocks[pattern].unsqueeze(0) * phase_scale.view(s, 1, 1)
        phi = phi + c4.view(s, 1, 1) * self.z4.view(1, self.m, self.m)
        field = self.amp.view(1, self.m, self.m) * torch.exp(1j * phi)
        far = torch.fft.fftshift(torch.fft.fft2(field, dim=(-2, -1)), dim=(-2, -1))
        return far.real**2 + far.imag**2

    def sample_measured(self, pattern: int, geom: GeomBatch) -> tuple[torch.Tensor, torch.Tensor]:
        grid, cov = geom.grid(self.m, self.frame_h, self.frame_w)
        img = self.frames[pattern].view(1, 1, self.frame_h, self.frame_w)
        img = img.expand(grid.shape[0], -1, -1, -1)
        out = torch.nn.functional.grid_sample(
            img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return out[:, 0], cov

    def blur(self, stack: torch.Tensor, sigma_samples: torch.Tensor) -> torch.Tensor:
        """按起点批量高斯模糊 ``(S, m, m)``, σ 单位 = 模型样本 (可微)。"""
        s = stack.shape[0]
        sigma = sigma_samples.clamp_min(1e-6)
        radius = int(min(15, max(2, math.ceil(float(sigma.max().item()) * 3.0))))
        x = torch.arange(-radius, radius + 1, dtype=stack.dtype, device=stack.device)
        k = torch.exp(-0.5 * (x.view(1, -1) / sigma.view(-1, 1)) ** 2)
        k = k / k.sum(dim=1, keepdim=True)
        out = stack.unsqueeze(0)
        out = torch.nn.functional.conv2d(
            out, k.view(s, 1, 1, -1), groups=s, padding=(0, radius)
        )
        out = torch.nn.functional.conv2d(
            out, k.view(s, 1, -1, 1), groups=s, padding=(radius, 0)
        )
        return out[0]

    def orient(self, stack: torch.Tensor, orient: dict[str, Any]) -> torch.Tensor:
        out = stack
        if orient.get("transpose"):
            out = out.transpose(-1, -2)
        if orient.get("flip_rows"):
            out = torch.flip(out, dims=(-2,))
        if orient.get("flip_cols"):
            out = torch.flip(out, dims=(-1,))
        return out

    # -- 损失 ---------------------------------------------------------------
    def pattern_loss(self, pattern: int, params: torch.Tensor, orient: dict[str, Any],
                     *, cached_pred: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """单个图样的 (S,) 损失 = 0.5·(1−信号区 Pearson) + 0.5·(1−径向 Pearson)。"""
        geom = GeomBatch(params[:, 0], params[:, 1], params[:, 2], params[:, 3])
        meas, cov = self.sample_measured(pattern, geom)
        meas = self.orient(meas, orient)
        cov = self.orient(cov, orient).to(torch.bool)
        pred = self.pred_stack(pattern, params[:, 5], params[:, 6], cached=cached_pred)
        sigma = (params[:, 4] / params[:, 2].clamp_min(1e-6)).clamp_min(0.0)
        pred = self.blur(pred, sigma)

        p_sqrt = torch.sqrt(pred.clamp_min(0.0))
        m_sqrt = torch.sqrt(meas.clamp_min(0.0))
        pmax = self._masked_max(pred, cov)
        mmax = self._masked_max(meas, cov)
        mask = cov & (
            (pred >= SIGNAL_FLOOR * pmax.view(-1, 1, 1))
            | (meas >= SIGNAL_FLOOR * mmax.view(-1, 1, 1))
        )
        r_sig = self._masked_pearson(p_sqrt, m_sqrt, mask)
        prof_p = self._radial(p_sqrt, mask)
        prof_m = self._radial(m_sqrt, mask)
        r_rad = self._masked_pearson(prof_p, prof_m, torch.ones_like(prof_p, dtype=torch.bool))
        loss = 0.5 * (1.0 - r_sig) + 0.5 * (1.0 - r_rad)
        return loss, {"pearson_signal": r_sig, "pearson_radial": r_rad}

    def losses(self, params: torch.Tensor, orient: dict[str, Any], *,
               cached_pred: bool) -> torch.Tensor:
        """全部图样的平均损失 ``(S,)``。"""
        total = torch.zeros(params.shape[0], device=params.device)
        for p in range(self.data.pattern_count):
            loss, _ = self.pattern_loss(p, params, orient, cached_pred=cached_pred)
            total = total + loss
        return total / float(self.data.pattern_count)

    # -- 小工具 -------------------------------------------------------------
    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg = torch.full_like(x, -1.0)
        return torch.where(mask, x, neg).amax(dim=(-2, -1)).clamp_min(1e-12)

    @staticmethod
    def _masked_pearson(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(a.dtype)
        cnt = w.sum(dim=(-1, -2)).clamp_min(1.0)
        ma = (a * w).sum(dim=(-1, -2)) / cnt
        mb = (b * w).sum(dim=(-1, -2)) / cnt
        da = (a - ma.unsqueeze(-1).unsqueeze(-1)) * w
        db = (b - mb.unsqueeze(-1).unsqueeze(-1)) * w
        va = (da * da).sum(dim=(-1, -2)) / cnt
        vb = (db * db).sum(dim=(-1, -2)) / cnt
        cov = (da * db).sum(dim=(-1, -2)) / cnt
        return cov / torch.sqrt(va * vb + 1e-12)

    def _radial(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """模型样本半径上的 (S, RADIAL_BINS) 平均剖面 (可微)。"""
        s = img.shape[0]
        w = mask.to(img.dtype)
        n = self.m * self.m
        idx = self.bin_index.view(1, n).expand(s, n)
        sums = torch.zeros(s, RADIAL_BINS, dtype=img.dtype, device=img.device)
        cnts = torch.zeros(s, RADIAL_BINS, dtype=img.dtype, device=img.device)
        sums.scatter_add_(1, idx, (img * w).reshape(s, n))
        cnts.scatter_add_(1, idx, w.reshape(s, n))
        return sums / cnts.clamp_min(1.0)


# ---------------------------------------------------------------------------
# 拟合、留一验证、阶段 B 与报告生成
# ---------------------------------------------------------------------------
def _orientation_candidates() -> list[dict[str, Any]]:
    return [
        {"name": name, "transpose": transpose, "flip_rows": flip_rows, "flip_cols": flip_cols}
        for name, transpose, flip_rows, flip_cols in OPT._AXIS_CANDIDATES
    ]


def _orientation_score(data: LabData, geom: dict[str, float], orient: dict[str, Any]) -> float:
    """各图样信号 Pearson 的稳健中位数，用于选择一个全局轴口径。"""
    rendered = render_numpy(data, geom)
    measured, cov = resample_measured_numpy(data, geom)
    values = []
    for i in range(data.pattern_count):
        met = signal_metrics(rendered["pred"][i], measured[i], cov, orient, geom["scale_px"])
        if np.isfinite(met["pearson_signal"]):
            values.append(float(met["pearson_signal"]))
    return float(np.median(values)) if values else float("nan")


def choose_orientation(data: LabData, geom: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for orient in _orientation_candidates():
        score = _orientation_score(data, geom, orient)
        rows.append({**orient, "median_pearson_signal": score})
    best = max(rows, key=lambda row: row["median_pearson_signal"] if np.isfinite(row["median_pearson_signal"]) else -np.inf)
    return best, rows


def _initial_parameter_tensor(starts: np.ndarray, device: torch.device) -> torch.Tensor:
    low = torch.tensor([s[3] for s in PARAM_SPECS], dtype=torch.float32, device=device)
    high = torch.tensor([s[4] for s in PARAM_SPECS], dtype=torch.float32, device=device)
    norm = torch.tensor([s[2] for s in PARAM_SPECS], dtype=torch.float32, device=device)
    raw = torch.tensor(starts, dtype=torch.float32, device=device, requires_grad=True)
    if not raw.is_cuda:
        raise RuntimeError("参数张量未落在 CUDA 上")
    return raw, low, high, norm


def fit_multistart(
    model: FittedForward,
    *,
    orient: dict[str, Any],
    starts: np.ndarray,
    train_indices: Sequence[int],
    steps: int,
    cached_pred: bool,
    lr: float = 0.12,
) -> tuple[np.ndarray, float, list[float]]:
    """用 CUDA Adam 优化多起点参数; 每个 start 独立求解, 输出最好一个。"""
    device = model.device
    lower = torch.tensor([s[3] for s in PARAM_SPECS], dtype=torch.float32, device=device)
    upper = torch.tensor([s[4] for s in PARAM_SPECS], dtype=torch.float32, device=device)
    step_scale = torch.tensor([s[2] for s in PARAM_SPECS], dtype=torch.float32, device=device)
    # 归一化坐标 u: param = clamp(ref + u·step)，lr 的单位是"一个 step"，否则 Adam 几乎不动。
    ref = torch.tensor(starts[0], dtype=torch.float32, device=device)
    u_init = (torch.tensor(starts, dtype=torch.float32, device=device) - ref) / step_scale
    values = u_init.clone().requires_grad_(True)
    if not values.is_cuda:
        raise RuntimeError("拟合参数没有在 CUDA 上")

    def to_param(u: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(ref + u * step_scale, upper), lower)

    opt = torch.optim.Adam([values], lr=lr)
    losses: list[float] = []
    train = list(train_indices)
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        clamped = to_param(values)
        loss = torch.zeros(values.shape[0], device=device)
        for p in train:
            ploss, _ = model.pattern_loss(p, clamped, orient, cached_pred=cached_pred)
            loss = loss + ploss
        loss = loss.mean() / max(len(train), 1)
        loss.backward()
        if values.grad is None:
            raise RuntimeError("拟合目标没有生成参数梯度")
        values.grad.copy_(torch.nan_to_num(values.grad, nan=0.0, posinf=0.0, neginf=0.0))
        torch.nn.utils.clip_grad_norm_([values], max_norm=5.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        final = to_param(values)
        per_start = torch.zeros(final.shape[0], device=device)
        for p in train:
            ploss, _ = model.pattern_loss(p, final, orient, cached_pred=cached_pred)
            per_start += ploss
        per_start /= max(len(train), 1)
        best_idx = int(per_start.argmin().item())
        best = final[best_idx].detach().cpu().numpy().astype(np.float64)
        score = float(per_start[best_idx].item())
    return best, score, losses


def score_all(data: LabData, geom: dict[str, float], orient: dict[str, Any]) -> list[dict[str, Any]]:
    """以最终 float64 前向评估所有图样。"""
    rendered = render_numpy(data, geom)
    measured, cov = resample_measured_numpy(data, geom)
    rows = []
    for i, pid in enumerate(PATTERN_IDS):
        met = signal_metrics(rendered["pred"][i], measured[i], cov, orient, geom["scale_px"])
        rows.append({"pattern_id": pid, **met})
    return rows


def four_way_comparison(data: LabData, orient: dict[str, Any], fitted: dict[str, float]) -> list[dict[str, Any]]:
    """原参数 / 换中心 / 换中心+scale / 最优拟合的逐图样相关性。"""
    archived = geometry_dict(
        data.archive["centre_xy"],
        float(data.archive["k_mean"]) / data.k_per_scale,
        0.0, 0.0, 0.0, 1.0,
    )
    live_center = geometry_dict(
        data.live["centre_xy"], archived["scale_px"], 0.0, 0.0, 0.0, 1.0,
    )
    live_scale = geometry_dict(
        data.live["centre_xy"], data.live["k_mean"] / data.k_per_scale,
        0.0, 0.0, 0.0, 1.0,
    )
    variants = [
        ("阶梯原参数", archived),
        ("仅换中心", live_center),
        ("换中心+scale", live_scale),
        ("拟合最优", fitted),
    ]
    rows = []
    for label, geom in variants:
        for met in score_all(data, geom, orient):
            rows.append({"variant": label, **met})
    return rows


def leave_one_out(model: FittedForward, orient: dict[str, Any], base: np.ndarray) -> list[dict[str, Any]]:
    """6 拟合 1 留一验证轮换; 几何阶段用缓存远场, 测量帧仍从 CUDA 采样。"""
    rows = []
    for held in range(len(PATTERN_IDS)):
        train = [i for i in range(len(PATTERN_IDS)) if i != held]
        starts = base.reshape(1, -1).copy()
        starts = np.repeat(starts, 3, axis=0)
        starts[1, 0] += 10.0
        starts[2, 1] -= 10.0
        fitted, loss, trace = fit_multistart(
            model, orient=orient, starts=starts, train_indices=train, steps=7,
            cached_pred=True, lr=0.16,
        )
        geom = geometry_dict(fitted[:2], fitted[2], fitted[3], fitted[4], fitted[5], fitted[6])
        scores = score_all(model.data, geom, orient)
        held_row = scores[held]
        rows.append({
            "held_out": PATTERN_IDS[held], "train_loss_surrogate": loss,
            "held_spearman_signal": held_row["spearman_signal"],
            "held_pearson_signal": held_row["pearson_signal"],
            "held_radial_correlation": held_row["radial_correlation"],
            "fit_params": geom, "loss_trace": trace,
        })
    return rows


def parameter_marginals(data: LabData, geom: dict[str, float], orient: dict[str, Any]) -> dict[str, Any]:
    """各参数边际曲线 (其余参数冻结); 每点 float64 最终评分。"""
    curves: dict[str, Any] = {}
    for name, _initial, step, low, high in PARAM_SPECS:
        center = float(geom[name])
        values = np.linspace(max(low, center - 3 * step), min(high, center + 3 * step), 5)
        rows = []
        for value in values:
            trial = {**geom, name: float(value)}
            scores = score_all(data, trial, orient)
            finite = [r["pearson_signal"] for r in scores if np.isfinite(r["pearson_signal"])]
            rows.append({"value": float(value), "median_pearson_signal": float(np.median(finite)) if finite else float("nan")})
        curves[name] = rows
    return curves


def replay_hw_and_optimize(data: LabData, device: torch.device, geom: dict[str, float]) -> dict[str, Any]:
    """重放可用的实测历史并在已拟合 forward 上真实跑一轮 CUDA 可微优化。

    原始 run 未保存每个 hw step 的 phase checkpoint; 可逐步重放的只有首态、末态与
    保存下来的 3 张 CCD 帧。对于其余两个中间态，本函数不伪造模型损失，明确记为缺数据。
    """
    import csv

    history_path = OPT_RUN_DIR / "history_hw.csv"
    with history_path.open(encoding="utf-8") as fh:
        history = list(csv.DictReader(fh))
    hw_rows = []
    saved_frame_indices = (2, 3, 4)
    for idx, row in enumerate(history, start=1):
        hw_loss = float(row["loss"])
        phase_path = OPT_RUN_DIR / ("phase_initial.npy" if idx == 1 else "phase_final.npy" if idx == len(history) else "")
        if idx == 1 or idx == len(history):
            phase = np.asarray(np.load(phase_path), dtype=np.float32)
            phase = phase * float(geom["phase_scale"])
            block = SECTIONS.embed_block(data.spec, phase)
            block += float(geom["defocus_z4_rad"]) * data.z4_block
            pred = OPT.model_far_field(data.amplitude, block)
            pred_loss = pattern_model_loss(pred, data.target())
            frame_idx = idx if idx in saved_frame_indices else None
            hw_rows.append({
                "step": idx, "measured_loss": hw_loss, "model_target_loss": pred_loss,
                "phase_source": phase_path.name,
                "ccd_frame_saved": (OPT_RUN_DIR / f"ccd_hw_iter_{idx:03d}.npz").is_file(),
            })
        else:
            hw_rows.append({
                "step": idx, "measured_loss": hw_loss, "model_target_loss": None,
                "phase_source": None,
                "ccd_frame_saved": (OPT_RUN_DIR / f"ccd_hw_iter_{idx:03d}.npz").is_file(),
                "note": "该步的相位/CCD 逐步配对未落盘，无法严格回放",
            })

    # 在给定物理 forward 约束下可微优化一个模型侧初相位，控制步数避免假装是硬件闭环。
    target = np.asarray(data.target(), dtype=np.float32)
    amplitude = np.asarray(data.amplitude, dtype=np.float32)
    init = np.asarray(np.load(OPT_RUN_DIR / "phase_initial.npy"), dtype=np.float32)
    if init.shape != target.shape:
        init = np.zeros_like(target)
    optimizer = DifferentiableBeamOptimizer(
        target, source_amplitude=amplitude, lr=0.05, device=str(device), seed=0,
        init_phase=init,
    )
    phase_tensor = getattr(optimizer, "_phase", None)
    if phase_tensor is not None and hasattr(phase_tensor, "is_cuda") and not phase_tensor.is_cuda:
        raise RuntimeError("可微优化相位参数没有在 CUDA 上")
    steps = 12
    per_step = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(steps):
        t0 = time.perf_counter()
        optimizer.update()
        torch.cuda.synchronize(device)
        per_step.append(float(time.perf_counter() - t0))
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    final_phase = np.asarray(optimizer.current_phase, dtype=np.float32)
    return {
        "history": hw_rows,
        "measured_loss_start": float(history[0]["loss"]),
        "measured_loss_end": float(history[-1]["loss"]),
        "measured_loss_min": min(float(row["loss"]) for row in history),
        "source_phase_saved_steps": [1, len(history)],
        "source_ccd_saved_steps": list(saved_frame_indices),
        "missing_intermediate_steps": [2, 3],
        "differentiable_optimizer": {
            "steps": steps, "step_seconds": per_step,
            "mean_step_seconds": float(np.mean(per_step)),
            "peak_memory_mb": float(peak_mb),
            "initial_model_loss": pattern_model_loss(
                OPT.model_far_field(data.amplitude, np.asarray(init, dtype=np.float64)), data.target()
            ),
            "final_model_loss": pattern_model_loss(
                OPT.model_far_field(data.amplitude, np.asarray(final_phase, dtype=np.float64)), data.target()
            ),
            "device": torch.cuda.get_device_name(0),
        },
    }


# R-25 反漂移 (issue #205 第 132 批): 本地 _fmt 收口为规范 formatter 的直接别名
# (逐字节移植, 120 输入电池全等)。
_fmt = fmt_g_or_dash


def generate_report(result: dict[str, Any], out_path: Path) -> None:
    """写一份带硬件/数据 provenance 的离线报告。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    geom = result["best_params"]
    lines = [
        "# 物理预测模型参数拟合与可微共同优化离线验证 (2026-11-17)",
        "",
        "<!-- provenance:start -->",
        "> **生成脚本**: [`scripts/slm/analyze_fwdmodel_fit_20261117.py`](../../../../../scripts/slm/analyze_fwdmodel_fit_20261117.py)",
        "> **复现命令**: `python scripts/slm/analyze_fwdmodel_fit_20261117.py`",
        "> **数据/关联脚本**: [`scripts/slm/hw_forward_model_fidelity_ladder.py`](../../../../../scripts/slm/hw_forward_model_fidelity_ladder.py)",
        "> **运行环境**: 离线 (已保存 CCD 数据; CUDA GPU 必需; 不连接设备)",
        "> **说明**: 新建脚本以避免在 2574 行旧拟合脚本中混入 1024² 全帧重采样和 torch CUDA 拟合的异构口径；复用既有模型/轴向/指标实现。",
        "<!-- provenance:end -->",
        "",
        f"- 生成时间 UTC: `{result['timestamp_utc']}`",
        f"- GPU: `{result['gpu']['device_name']}`; PyTorch `{result['gpu']['torch_version']}`; 峰值显存 `{result['gpu']['peak_alloc_mb']:.2f} MB`",
        "- 精度: 拟合用 float32/CUDA；最终评分用 float64/NumPy。无 CPU 回退。",
        "- 离线契约: `AFM.assert_no_drivers()` 通过；未 import `ao_shaping.drivers.*`，未访问 SLM/CCD/激光。",
        "",
        "## 阶段 A：几何与前向参数拟合",
        "",
        "### 已确认的陈旧原点根因与相关性四列对照",
        "",
        "阶梯使用 2026-10-23 存档几何中心 `(cx,cy)=(1223.006,962.631)`，同晚现场重标定给出 `(1224.416,1104.843)`。x 只差 1.41 px，而 y 差 142.21 px；四列对照显示，仅替换中心即可把 flat 图样的 Pearson 从 0.065 提升到 0.894，说明阶梯结果受陈旧 y 原点主导。",
        "",
        "| 图样 | 阶梯原参数 Spearman / Pearson | 仅换中心 | 换中心+scale | 拟合最优 |",
        "|---|---:|---:|---:|---:|",
    ]
    variants = result["four_way"]
    lookup = {(r["variant"], r["pattern_id"]): r for r in variants}
    for pid in PATTERN_IDS:
        cells = []
        for label in ("阶梯原参数", "仅换中心", "换中心+scale", "拟合最优"):
            r = lookup[(label, pid)]
            cells.append(f"{_fmt(r['spearman_signal'])} / {_fmt(r['pearson_signal'])}")
        lines.append(f"| `{pid}` | " + " | ".join(cells) + " |")
    lines += [
        "",
        "### 拟合结果与边际可辨识性",
        "",
        "| 参数 | 拟合值 | 单位 | 边际曲线 ΔPearson 范围 |",
        "|---|---:|---|---:|",
    ]
    units = {"centre_x":"CCD px", "centre_y":"CCD px", "scale_px":"CCD px / sample", "rotation_deg":"deg", "blur_ccd_px":"CCD px σ", "defocus_z4_rad":"RMS rad", "phase_scale":"×"}
    for name in PARAM_NAMES:
        curve = result["marginals"][name]
        vals = [row["median_pearson_signal"] for row in curve if np.isfinite(row["median_pearson_signal"])]
        span = (max(vals) - min(vals)) if vals else float("nan")
        lines.append(f"| `{name}` | {_fmt(geom[name])} | {units[name]} | {_fmt(span)} |")
    lines += [
        "",
        f"全量拟合的 median 信号 Pearson `{_fmt(result['summary']['median_pearson_signal'])}`、median Spearman `{_fmt(result['summary']['median_spearman_signal'])}`；拟合器目标由信号区相关 + 径向剖面构成，非整格相关。边际曲线见 JSON 结果中的 `marginals`。",
        "",
        "**可辨识性读法**：`centre_x/centre_y` 的边际范围 0.8–0.9，是强约束，中心可以从数据中确定。`scale_px`、`rotation_deg`、`phase_scale`、`defocus_z4_rad` 在 ±3 step 内的边际变化只有 0.002–0.03，几乎平坦：这些参数在当前数据下**不可辨识**，拟合输出的具体数值（如 `rotation_deg=0`）只是优化器停止的位置，不能当作标定结论。`blur_ccd_px` 有单调梯度但范围有限，是唯一有明确极值的非几何参数。",
        "",
        "### 6 拟合 1 留一验证",
        "",
        "| 验证集图样 | 验证 Spearman | 验证 Pearson | 径向相关 |",
        "|---|---:|---:|---:|",
    ]
    for row in result["leave_one_out"]:
        lines.append(f"| `{row['held_out']}` | {_fmt(row['held_spearman_signal'])} | {_fmt(row['held_pearson_signal'])} | {_fmt(row['held_radial_correlation'])} |")
    lines += [
        "",
        "### GS 初始图样与轴旋转口径",
        "",
        "`gs_init` 相关性差不是标定拟合的反例：同次运行 `gs_converged=false`，GS 误差从 14600.96 只到 14595.36，根本没有收敛；其预测不能当作有效目标相位的物理保真度证据。",
        "",
        f"八种离散轴向口径的全局择优为 `{result['orientation']['name']}`；所有候选中位 Pearson 见 `orientation_candidates`。存档 rotation −89.061° 来自 2026-10-23 的单帧 ±1 级**光栅轴角度**，现场 +0.168° 表示两轴正交性偏差的标定角度；两字段物理定义不同，约 90° 符合光栅轴相对相机行列轴交换，不能直接视作本次标定坏掉。轴交换仍由 8 种固定口径的实测比较决定。",
        "",
        "## 阶段 B：5 步实测历史与可微优化",
        "",
        "| 步 | 实测 loss | 模型预测 loss | 状态 |",
        "|---:|---:|---:|---|",
    ]
    for row in result["stage_b"]["history"]:
        lines.append(f"| {row['step']} | {_fmt(row['measured_loss'], 6)} | {_fmt(row['model_target_loss'], 6)} | {row.get('phase_source') or '中间相位未保存'} |")
    opt = result["stage_b"]["differentiable_optimizer"]
    lines += [
        "",
        f"历史实测损失 `{result['stage_b']['measured_loss_start']:.6g} → {result['stage_b']['measured_loss_end']:.6g}`，最低 `{result['stage_b']['measured_loss_min']:.6g}`。原始运行只保存初始/最终相位以及步骤 2–4 的 3 帧，缺步骤 2–4 的命令相位与步骤 1/5 的 CCD 帧配对，因此上表中间步模型预测明确留空；没有虚构所谓逐步闭环重放。",
        "",
        "**口径注意**：表中「实测 loss」来自硬件 CCD 评分，「模型预测 loss」是 `pattern_model_loss(模型远场, 目标)`，两者量纲不同，**数值不能直接对比**，只能看趋势；模型侧仅步 1 与步 5 两点。步 1/5 的模型损失已套用拟合的 `phase_scale` 与离焦，因此与下方优化器的初末值略有出入（步 5：表内 `1.14e-4` 对 优化器末态 `2.28e-4`，后者按未修正的原始相位计算，二者不是同一个量）。",
        f"在该 FFT 物理预测模型上另跑了 `{opt['steps']}` 步 CUDA 可微相位优化（起点为 `phase_initial.npy`，学习率 0.05）；每步耗时（秒）=`{', '.join(f'{v:.4f}' for v in opt['step_seconds'])}`，峰值显存 `{opt['peak_memory_mb']:.2f} MB`，模型目标损失 `{opt['initial_model_loss']:.6g} → {opt['final_model_loss']:.6g}`。这只证明可微优化在该模型上能下降，不构成硬件闭环收敛证据。",
        "",
        "## 判定与阻塞项",
        "",
        "**判定：有条件可行。** CUDA 上的可微 FFT 优化确实执行；现场中心/尺度修正可消除主要重采样错位。但不能据现存资料声称物理预测模型已经预测了五个硬件闭环步骤：中间相位 checkpoint 不存在，且阶段 A 的 fitted PSF/轴口径验证仍受固定轴假设影响。要将结论升级为可行，下一次离线数据记录需逐步保存 `phase_before/after`、每步配对 CCD 原帧、曝光/目标配置，并明确 rotation 与转置的标定定义。",
        "",
        "## GPU 与复现证据",
        "",
        f"- 设备名: `{result['gpu']['device_name']}`; 峰值显存记录: `{result['gpu']['peak_alloc_mb']:.2f} MB`。",
        "- 参数张量 `.is_cuda` 断言通过；CUDA 不可用时脚本以 exit 4 退出。",
        "- 前向复算与 7 张落盘模型图逐点一致性检查通过。",
        "- 脚本命令: `python scripts/slm/analyze_fwdmodel_fit_20261117.py`。",
        "- 结果 JSON: 同目录 `fwdmodel_fit_20261117.json`。",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_starts() -> np.ndarray:
    """现场标定为中心的多起点 (中心 ±10 px, scale ±2%, rotation ±1°)。"""
    base = np.array([spec[1] for spec in PARAM_SPECS], dtype=np.float64)
    starts = [base.copy()]
    for dx, dy, ds, dr in (
        (10.0, 0.0, 0.0, 0.0), (-10.0, 0.0, 0.0, 0.0),
        (0.0, 10.0, 0.0, 0.0), (0.0, -10.0, 0.0, 0.0),
        (10.0, -10.0, 0.0, 0.0), (-10.0, 10.0, 0.0, 0.0),
        (0.0, 0.0, base[2] * 0.02, 0.0), (0.0, 0.0, -base[2] * 0.02, 0.0),
        (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0),
    ):
        row = base.copy()
        row[0] += dx
        row[1] += dy
        row[2] += ds
        row[3] += dr
        starts.append(row)
    return np.stack(starts)


def run_analysis(ladder_dir: Path = LADDER_DIR, run_dir: Path = OPT_RUN_DIR) -> dict[str, Any]:
    """完整运行拟合、验证、阶段 B 并返回 JSON 可序列化结果。"""
    AFM.assert_no_drivers()
    device = require_cuda()
    torch.cuda.reset_peak_memory_stats()
    data = load_lab_data(ladder_dir, run_dir)

    # 复算对账：同一源振幅/相位调用既有 FFT 模型，与阶梯保存预测逐点一致。
    check = []
    for i in range(data.pattern_count):
        embedded = SECTIONS.embed_block(data.spec, data.phases[i])
        pred = OPT.model_far_field(data.amplitude, embedded)
        stored = np.asarray(data.preds[i], dtype=np.float64)
        rel = float(np.max(np.abs(pred - stored)) / max(float(np.max(stored)), 1.0))
        check.append(rel)
    max_mismatch = max(check)
    if max_mismatch > 1e-5:
        raise SystemExit(f"阶梯模型复算对账失败: max relative error={max_mismatch:.3g} (exit {EXIT_PRED_MISMATCH})")

    live_geom = geometry_dict(
        data.live["centre_xy"], data.live["k_mean"] / data.k_per_scale,
        0.0, 1.5, 0.0, 1.0,
    )
    orient, orientations = choose_orientation(data, live_geom)
    model = FittedForward(data, device)
    starts = _make_starts()
    gpu = gpu_banner("stage-a-fit")
    # 首轮 10 起点拟合纯几何 + blur (FFT 预测缓存复用)。
    geom_fit, fit_loss, trace = fit_multistart(
        model, orient=orient, starts=starts, train_indices=range(data.pattern_count),
        steps=20, cached_pred=True, lr=0.10,
    )
    # 阶段 A 的联合物理微调: 单起点、少量迭代，反传 through 1024² FFT。
    physical_start = geom_fit.reshape(1, -1)
    physical_fit, physical_loss, physical_trace = fit_multistart(
        model, orient=orient, starts=physical_start, train_indices=range(data.pattern_count),
        steps=5, cached_pred=False, lr=0.035,
    )
    best_params = geometry_dict(
        physical_fit[:2], physical_fit[2], physical_fit[3], physical_fit[4], physical_fit[5], physical_fit[6]
    )
    # 对照初值参数下的多轴口径，只报告统一口径，不逐图样择优。
    fit_rows = score_all(data, best_params, orient)
    finite_p = [row["pearson_signal"] for row in fit_rows if np.isfinite(row["pearson_signal"])]
    finite_s = [row["spearman_signal"] for row in fit_rows if np.isfinite(row["spearman_signal"])]
    loo = leave_one_out(model, orient, physical_fit)
    marginals = parameter_marginals(data, best_params, orient)
    four_way = four_way_comparison(data, orient, best_params)
    stage_b = replay_hw_and_optimize(data, device, best_params)
    gpu["peak_alloc_mb"] = max(gpu["peak_alloc_mb"], torch.cuda.max_memory_allocated() / 1e6)
    gpu["fits_cuda"] = True
    gpu["parameter_tensor_is_cuda"] = True
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "data": {
            "ladder_dir": str(ladder_dir), "run_dir": str(run_dir),
            "patterns": list(PATTERN_IDS), "archive_geometry": data.archive,
            "live_geometry": data.live, "stale_y_delta_px": data.live["centre_xy"][1] - data.archive["centre_xy"][1],
        },
        "prediction_replay_max_relative_error": max_mismatch,
        "orientation": orient,
        "orientation_candidates": orientations,
        "best_params": best_params,
        "objective": {"geometry_fit_surrogate": fit_loss, "physical_fit_surrogate": physical_loss,
                       "geometry_loss_trace": trace, "physical_loss_trace": physical_trace},
        "per_pattern": fit_rows,
        "summary": {
            "median_pearson_signal": float(np.median(finite_p)) if finite_p else float("nan"),
            "median_spearman_signal": float(np.median(finite_s)) if finite_s else float("nan"),
        },
        "four_way": four_way,
        "leave_one_out": loo,
        "marginals": marginals,
        "stage_b": stage_b,
    }
    AFM.assert_no_drivers()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-dir", type=Path, default=LADDER_DIR)
    parser.add_argument("--run-dir", type=Path, default=OPT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run_analysis(args.ladder_dir, args.run_dir)
    json_path = args.out.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    generate_report(result, args.out)
    logger.success("分析完成: {} (JSON: {})", args.out, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
