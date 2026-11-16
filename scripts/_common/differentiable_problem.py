"""可微分光束整形「问题重建」的唯一共享入口 (离线与硬件脚本共用)。

从一次硬件运行目录 (``data/debug/differentiable_opt/<run>/``) 重建纯模型侧
的可微分优化问题: 读 ``run_config.json`` 的 ``params`` / ``geometry``, 装配
模型网格 (:class:`ModelSpec`)、源面振幅、目标远场强度与保存的初始相位
``phase_initial.npy``。重建口径**逐行对齐**
``scripts/slm/hw_differentiable_phase_opt.py::run_analyze_only`` —— 同一份
代码同时被离线学习率扫描 (``scripts/slm/sweep_differentiable_lr.py``) 与
硬件脚本的离线重分析消费, 避免两处各写一份 bootstrap 而悄悄漂移。

本模块**导入时零设备依赖**: 顶层只 import 标准库与 numpy;
``hw_differentiable_phase_opt`` 按路径延迟加载 (其 dataclass 要求先把模块
注册进 ``sys.modules`` 再 ``exec_module``), torch / 优化器类同样延迟导入。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover – 仅类型
    from ao_shaping.algorithm.signal_processing.differentiable_beam import (
        DifferentiableBeamOptimizer,
    )

#: 仓库根 (scripts/_common/x.py -> parents[2])。
ROOT = Path(__file__).resolve().parents[2]

#: 被延迟加载的硬件脚本 (问题重建口径的事实来源)。
_HW_MODULE_NAME = "hw_differentiable_phase_opt"
_HW_MODULE_PATH = ROOT / "scripts" / "slm" / "hw_differentiable_phase_opt.py"


def _ensure_sys_path() -> None:
    """把仓库根 / src / scripts 放进 ``sys.path`` (幂等)。

    ``hw_differentiable_phase_opt`` 顶层 import ``scripts._common.*`` 与
    ``scripts.slm.*``, 所以仓库根本身必须可 import —— 与硬件脚本自己的
    bootstrap 保持同一套路径。
    """
    for entry in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def load_hw_module() -> Any:
    """按路径加载 ``hw_differentiable_phase_opt`` 模块 (进程内缓存)。

    Returns:
        加载完成的模块对象 (含 ``ModelSpec`` / ``build_model_spec`` /
        ``build_source_amplitude`` / ``build_target`` 等重建口径事实来源)。

    Raises:
        ImportError: 找不到硬件脚本或其 loader 不可用。
    """
    _ensure_sys_path()
    existing = sys.modules.get(_HW_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(_HW_MODULE_NAME, _HW_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法按路径加载硬件脚本: {_HW_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    # 必须先注册再 exec: 模块里的 @dataclass 装饰器在 exec 期间解析
    # ``sys.modules[__name__]``, 缺了这步会报
    # "'NoneType' object has no attribute '__dict__'"。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class DifferentiableProblem:
    """从一次运行目录重建的可微分优化问题 (纯模型侧, 不含任何设备)。

    Attributes:
        spec: 模型网格几何 (``hw_differentiable_phase_opt.ModelSpec``;
            该类由延迟加载的模块定义, 故此处注解为 ``Any``)。
        amplitude: 源面 (瞳面) 振幅, ``M×M`` float64, 口径外为零。
        target: 目标远场强度, ``M×M`` float64 (未归一化, 优化器内部会做
            峰值归一化)。
        target_info: ``build_target`` 返回的描述字典 (``side_samples`` /
            ``side_ccd_px`` / ``peak_sample`` / ``note`` 等)。
        k_mean: 几何标定给出的各向同性远场标度 (CCD 像元/面板像元)。
        centre: 远场网格直流格下标 ``(c, c)`` (目标中心所在格)。
        init_phase: 运行目录保存的初始相位 (弧度, ``N×N`` 或 ``M×M``);
            目录里没有 ``phase_initial.npy`` 时为 ``None`` (优化器将按
            ``seed`` 随机抽一个)。
        params: ``run_config.json`` 的 ``params`` 段原样 (``lr`` / ``iters`` /
            ``seed`` / ``target`` / ``target_side_px`` 等)。
        geometry: ``run_config.json`` 的 ``geometry`` 段原样。
        run_dir: 运行目录路径。
    """

    spec: Any
    amplitude: npt.NDArray[np.floating]
    target: npt.NDArray[np.floating]
    target_info: dict[str, Any]
    k_mean: float
    centre: tuple[int, int]
    init_phase: npt.NDArray[np.floating] | None
    params: dict[str, Any]
    geometry: dict[str, Any]
    run_dir: Path


def load_problem_from_run_dir(run_dir: Path) -> DifferentiableProblem:
    """从运行目录重建可微分优化问题 (口径 = ``run_analyze_only``)。

    Args:
        run_dir: 含 ``run_config.json`` 的运行目录
            (``phase_initial.npy`` 可选)。

    Returns:
        重建好的 :class:`DifferentiableProblem`。

    Raises:
        FileNotFoundError: ``run_config.json`` 缺失。
        ValueError: ``geometry.k_mean`` 缺失或非正 (无法把目标边长从 CCD
            像元换算到远场采样格)。
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"找不到运行配置: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    params: dict[str, Any] = cfg.get("params", {})
    geometry: dict[str, Any] = cfg.get("geometry", {})
    k_mean = float(geometry.get("k_mean", 0.0))
    if not np.isfinite(k_mean) or k_mean <= 0.0:
        raise ValueError(f"运行配置里没有可用的 K (几何标定缺失): {cfg_path}")

    hw = load_hw_module()
    pupil_center = tuple(float(v) for v in params.get("pupil_center", (960.0, 600.0)))
    spec = hw.build_model_spec(
        pupil_radius=float(params.get("pupil_radius", 480.0)),
        pupil_center=pupil_center,
        n_pupil=int(params.get("model_pupil_samples", 256)),
        padding=int(params.get("model_padding", 4)),
        w0_px=float(params.get("beam_w0_px", 260.0)),
    )
    amplitude = hw.build_source_amplitude(spec)
    target, target_info = hw.build_target(
        spec,
        str(params.get("target", "square")),
        float(params.get("target_side_px", 60.0)),
        k_mean,
        delta_spot_px=float(params.get("delta_spot_px", 15.0)),
    )
    init_path = run_dir / "phase_initial.npy"
    init_phase = np.load(init_path) if init_path.is_file() else None
    c = int(spec.centre_index)
    return DifferentiableProblem(
        spec=spec,
        amplitude=amplitude,
        target=target,
        target_info=target_info,
        k_mean=k_mean,
        centre=(c, c),
        init_phase=init_phase,
        params=params,
        geometry=geometry,
        run_dir=run_dir,
    )


def build_optimizer(
    problem: DifferentiableProblem,
    *,
    lr: float,
    device: str,
    seed: int | None = None,
    init_phase: npt.NDArray[np.floating] | None = None,
) -> DifferentiableBeamOptimizer:
    """按重建好的问题装配一个 :class:`DifferentiableBeamOptimizer`。

    Args:
        problem: :func:`load_problem_from_run_dir` 的产物。
        lr: Adam 学习率。
        device: torch 设备字符串 (``"cuda"`` / ``"cpu"``)。
        seed: 随机数种子 (仅当没有初始相位、优化器需要随机抽取时有意义)。
        init_phase: 覆盖 ``problem.init_phase`` 的初始相位 (弧度);
            ``None`` 时用问题自带的 (与 ``run_analyze_only`` 一致)。

    Returns:
        构造完成、尚未迭代的优化器。
    """
    from ao_shaping.algorithm.signal_processing.differentiable_beam import (
        DifferentiableBeamOptimizer,
    )

    phase = problem.init_phase if init_phase is None else init_phase
    # 必须复制初始相位: ``DifferentiableBeamOptimizer.__init__`` 用
    # ``np.asarray(init_phase, dtype=np.float32)`` —— 当传入数组已是 float32
    # (``phase_initial.npy`` 正是) 时 asarray **不复制**, 随后
    # ``torch.from_numpy(...).to("cpu")`` 在 CPU 上也不复制, 于是 Adam 的
    # ``step()`` 会就地改写这块内存。若复用同一个 ``problem`` 连续构建多个
    # 优化器, 前一个变体的迭代会污染后一个的初始相位 (扫描里每个变体都从
    # 上一个的终态起步, 结果全错)。CUDA 路径因 ``.to("cuda")`` 会复制而幸免,
    # 但离线扫描/测试常在 CPU 上跑, 故这里显式复制一份独占的 float32 数组。
    phase_copy = None if phase is None else np.array(phase, dtype=np.float32, copy=True)
    return DifferentiableBeamOptimizer(
        problem.target,
        source_amplitude=problem.amplitude,
        lr=float(lr),
        device=device,
        seed=seed,
        init_phase=phase_copy,
    )
