"""迁移学习闭环桌面页的纯参数层：默认值、校验、会话目录与运行参数。

为什么单独一个模块
------------------
PySide6 桌面页 (``transfer_desktop/page.py``) 与 Streamlit 页
(``gui/slm/shaping_controls.py``) 必须把同一张表单映射成同一个
:class:`~ao_shaping.tools.slm.transfer.workflow.TransferSession`，否则两条入口
会对同一次实验给出不同的默认值、不同的拒绝顺序或不同的会话目录。本模块只用
标准库与 ``ao_shaping.tools.slm.transfer``，**不 import** Qt / Streamlit /
``shaping_controls`` / ``transfer_candidates``，因此没有服务器也能单测，也不会
把 Qt 拖进 ``import ao_shaping.gui.slm``。

口径来源（不要另立一套）
------------------------
- 表单默认值镜像 ``shaping_controls.TRANSFER_RECIPE``。那个模块 import
  Streamlit，桌面页不能依赖，所以默认值在这里逐字镜像；测试以字面量钉住它们
  （不 import 那个模块），改一处而漏另一处会被测试抓住。
- 校验顺序镜像 ``TransferMethod.validate``：torch → 相机类型 → 形状 →
  ``preflight.check_parameters``。
- 会话目录镜像 ``plan_transfer_session_dir``：``<root>/debug/slm_transfer/<ts>``，
  只规划、不创建。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ao_shaping.utils.io.timestamp import format_ts

if TYPE_CHECKING:
    from ao_shaping.tools.slm.transfer.workflow import TransferSession

__all__ = [
    "CAMERA_CHOICES",
    "SHAPE_CHOICES",
    "SHAPE_LABELS",
    "STAGE_LABELS",
    "TransferFormValues",
    "build_session",
    "effective_window_span",
    "plan_session_dir",
    "validate_form",
]

#: 相机后端选项，与 ``bench_settings.CAM_TYPE_OPTIONS`` 同序。迁移学习只对大恒
#: 相机做过实验验证，其余后端由 :func:`validate_form` 在启动前拒绝。
CAMERA_CHOICES: tuple[str, ...] = ("daheng", "miicam", "sim")

#: 目标形状选项（顺序即下拉/勾选的显示顺序）。镜像
#: ``utils.image.target.metrics.POLYGON_SHAPE_CHOICES``。
SHAPE_CHOICES: tuple[str, ...] = ("square", "rectangle", "triangle", "star")

#: 形状 → 中文名。镜像 ``transfer_candidates.TRANSFER_SHAPE_LABELS``（该模块
#: import Streamlit，桌面页不能依赖）。
SHAPE_LABELS: dict[str, str] = {
    "square": "正方形",
    "rectangle": "长方形",
    "triangle": "三角形",
    "star": "五角星",
}

#: ``run_session`` 发出的阶段名 → 操作员读得懂的标签。镜像
#: ``shaping_controls.TRANSFER_STAGE_LABELS``（与 ``transfer_progress.STAGE_LABELS``
#: 是同一份内容）；测试直接比较两者，防止漂移。
STAGE_LABELS: dict[str, str] = {
    "capture": "当前光路采集（探针 + 逐帧自动曝光）",
    "reuse_capture": "复用已有采集",
    "nominal_initialization": "名义模型初始化（冷启动，未标定）",
    "fit": "物理预测模型迁移 / 拟合",
    "CCD_model_blocks": "CCD 模型配对块",
    "exposure_response_capture": "曝光阶梯诊断",
    "exposure_response_analysis": "曝光响应分析",
    "blur_calibration": "有效展宽 / 辐射尺度校准",
    "CCD_anchored_blocks": "CCD 锚定优化（验收：A–B–B–A 或仅模型损失）",
    "GS_and_differentiable_synthesis": "GS + 可微相位合成（离线）",
    "transferred_phase_differentiable_synthesis": (
        "历史接受相位迁移 + 可微合成（离线）"
    ),
    "raw_data_audit": "原始数据审计",
    "summary": "指标摘要",
    "direction_check": "方向核对",
}


@dataclass(frozen=True)
class TransferFormValues:
    """桌面页表单的纯数据快照（与 Streamlit 的 ``values`` 字典一一对应）。

    字段顺序即表单顺序：目标 → 预算 → 相机 / SLM → 高级口径 → 历史产物路径。
    所有字段都带有限默认值（镜像 ``TRANSFER_RECIPE`` 与 Streamlit 控件的
    ``value=``），因此空表单也能直接构造。

    Attributes:
        shapes: 目标形状（``SHAPE_CHOICES`` 的子集）。
        side: 目标尺寸 (CCD px)。
        period: 载波周期 (panel px)。
        count: 分组随机探针数（必须是 6 的倍数）。
        iterations: 模型参数拟合迭代数。
        blocks: CCD 锚定短块数。
        updates: 每块内的模型更新次数。
        frames: 每个测量点的原始帧数。
        seed: 探针/初始化随机种子。
        camera: 相机后端（``CAMERA_CHOICES`` 之一）。
        cam_id: 相机序号。
        slm_number: SLM 序号。
        calibration: 跨形状校准开关。``True`` 映射成 ``TransferSession.calibration
            = None``（让管线自己判定），``False`` 显式关闭。
        adaptation: 迁移模式（``auto`` / ``alignment`` / ``full``）。
        objective: 目标函数（``shape`` / ``cv`` / ``formation``）。
        accept_metric: 验收指标（``loss`` / ``cv``）。
        seed_policy: 初始化策略（``strict`` / ``transfer``）。
        abba: 是否做 A–B–B–A 配对验收。
        adaptive_exposure: 块间是否重测曝光。
        save_dataset: 是否写出采集 npz（离线 ``--dataset`` 用）。
        device: 计算设备（``auto`` / ``cpu`` / ``cuda``）。
        checkpoint: 冻结检查点路径原文（空 = 冷启动），由
            :func:`build_session` 去空白后转成 ``Path``。
        seed_run: 历史接受相位目录原文（空 = 不用），由
            :func:`build_session` 去空白后转成 ``Path``。
        window_span: 采样窗**半跨**（px）。``0`` = 自动派生（保持小目标行为完全
            不变，见 :func:`effective_window_span`）；``>0`` = 显式指定。
    """

    shapes: tuple[str, ...] = ("square",)
    side: float = 80.0
    period: float = 64.0
    count: int = 24
    iterations: int = 60
    blocks: int = 4
    updates: int = 10
    frames: int = 3
    seed: int = 102319
    camera: str = "daheng"
    cam_id: int = 0
    slm_number: int = 1
    calibration: bool = True
    adaptation: str = "auto"
    objective: str = "shape"
    accept_metric: str = "loss"
    seed_policy: str = "transfer"
    abba: bool = False
    adaptive_exposure: bool = False
    save_dataset: bool = False
    device: str = "auto"
    checkpoint: str = ""
    seed_run: str = ""
    window_span: float = 0.0


def effective_window_span(side: float, window_span: float = 0.0) -> float:
    """采样窗**半跨**（px）：显式值优先，否则由目标边长派生。

    为什么必须派生
    --------------
    采样窗是 ``validate.run`` 里 ``recentre_window(zero_xy, half_span_px=...)``
    的半径：目标方斑整个落在窗内，两轴各再留 :data:`WINDOW_MARGIN_PX` 的边，
    窗才既框住目标又不压窗边。管线默认半跨 220 px 只够 424 px 的目标，而实测
    光路上 ``side=900`` 是跑通过的 —— 上限的唯一物理依据是 0 级到帧边的距离
    （页面「读取零级」量出来），不是这个默认值。

    于是：

    * ``window_span > 0``（显式，例如命令行已验证过的 930）→ 原样返回；
    * 否则 ``max(220, side/2 + 8)``：采样窗必须放下整个目标 + 两侧留白。
      ``side`` 大到接近帧边时，派生出的半跨恰好等于 0 级到帧边的距离，
      窗刚好不越界 —— 这正是「目标边长上限」的物理含义。

    小目标行为不变：``side <= 424`` 时结果仍是 220.0，与管线默认值逐位相同。

    Args:
        side: 目标边长（CCD px）。
        window_span: 显式半跨（px）；``0`` = 自动派生。

    Returns:
        采样窗半跨（px），**不是全宽**。
    """
    from ao_shaping.tools.slm.transfer.preflight import (
        HALF_SPAN_PX,
        WINDOW_MARGIN_PX,
    )

    if float(window_span) > 0:
        return float(window_span)
    return max(HALF_SPAN_PX, float(side) / 2.0 + WINDOW_MARGIN_PX)


def validate_form(values: TransferFormValues) -> tuple[list[str], list[str]]:
    """按 Streamlit 页的顺序做运行前校验。

    顺序固定（与 ``TransferMethod.validate`` 一致）：torch 未安装 → 相机类型
    不是大恒 → 没选形状 → ``preflight.check_parameters``。前三条任何一条不通过
    都直接返回：缺 torch 时讨论载波周期没有意义，也不会把一条"周期性"提示混进
    真正的阻塞原因里。

    Args:
        values: 表单快照。

    Returns:
        ``(errors, warnings)``。元素是可直接显示的中文文本；``preflight`` 的结论
        写成「哪里不对 + ``→`` 怎么改」两行。``errors`` 非空即禁止启动，
        ``warnings`` 只提示（例如未标定档的载波周期）。
    """
    torch_error = _torch_missing()
    if torch_error is not None:
        return [torch_error], []

    camera = str(values.camera)
    if camera != "daheng":
        return [
            f"迁移学习闭环目前只对大恒相机做过实验验证（相机类型 {camera!r}）。"
            "其他后端仅有接口支持；请在「相机类型」里选择 daheng"
        ], []

    if not values.shapes:
        return ["至少选择一个目标形状"], []

    from ao_shaping.tools.slm.transfer.preflight import check_parameters

    problems = check_parameters(
        side=float(values.side),
        period=float(values.period),
        count=int(values.count),
        iterations=int(values.iterations),
        blocks=int(values.blocks),
        updates=int(values.updates),
        frames=int(values.frames),
        shapes=tuple(values.shapes or ()),
        half_span_px=effective_window_span(float(values.side), values.window_span),
    )

    errors: list[str] = []
    warnings: list[str] = []
    for problem in problems:
        text = f"{problem.message}\n    → {problem.hint}"
        if problem.severity == "error":
            errors.append(text)
        else:
            warnings.append(text)
    return errors, warnings


def _torch_missing() -> str | None:
    """``validate_torch`` 的桌面副本：未安装 torch 时返回拒绝文本。"""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return f"迁移学习闭环需要 torch（未安装）: {exc}"
    return None


def plan_session_dir(root_dir: Path = Path("data")) -> Path:
    """规划（**不创建**）``<root>/debug/slm_transfer/<timestamp>`` 会话目录。

    ``run_session`` 用 ``exist_ok=False`` 创建会话目录（拒绝把两次运行的证据合
    并在一起是写入端约定），所以这里必须给一个尚不存在的路径。同一秒内双击两
    次时加 ``-2`` / ``-3`` 后缀，避免两次运行抢同一个目录。

    Args:
        root_dir: 数据根目录；默认 ``data``（相对当前工作目录）。

    Returns:
        计划使用的会话目录（其父目录也可能尚未创建）。
    """
    base = Path(root_dir) / "debug" / "slm_transfer"
    stamp = format_ts()
    candidate = base / stamp
    suffix = 2
    while candidate.exists():
        candidate = base / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def build_session(
    values: TransferFormValues, *, out: Path | None = None
) -> TransferSession:
    """表单值 → :class:`TransferSession`（与 Streamlit ``build_params`` 同映射）。

    只做映射，不调用 ``TransferSession.validate`` —— 拒绝路径是
    :func:`validate_form` 的职责，运行端 (``run_session``) 自己还会再校验一次。
    未在表单里出现的 ``TransferSession`` 字段（``initial_phase`` 等）保持管线
    默认值，与 Streamlit 一致；``window_span`` 例外——它是派生量，见
    :func:`effective_window_span`（``side`` 小的时候仍等于管线默认的 220）。

    Args:
        values: 表单快照。
        out: 会话目录。``None`` 时调用 :func:`plan_session_dir`（只规划）。

    Returns:
        可直接交给 ``run_session`` 的参数对象。
    """
    from ao_shaping.tools.slm.transfer.workflow import TransferSession

    checkpoint = str(values.checkpoint or "").strip()
    seed_run = str(values.seed_run or "").strip()
    return TransferSession(
        out=Path(out) if out is not None else plan_session_dir(),
        checkpoint=Path(checkpoint) if checkpoint else None,
        seed_run=Path(seed_run) if seed_run else None,
        shapes=tuple(values.shapes or ()),
        count=int(values.count),
        iterations=int(values.iterations),
        blocks=int(values.blocks),
        updates=int(values.updates),
        frames=int(values.frames),
        side=float(values.side),
        period=float(values.period),
        window_span=effective_window_span(float(values.side), values.window_span),
        seed=int(values.seed),
        camera=str(values.camera),
        cam_id=int(values.cam_id),
        slm_number=int(values.slm_number),
        calibration=None if values.calibration is True else False,
        adaptation=str(values.adaptation),
        accept_metric=str(values.accept_metric),
        objective=str(values.objective),
        seed_policy=str(values.seed_policy),
        abba=bool(values.abba),
        adaptive_exposure=bool(values.adaptive_exposure),
        save_dataset=bool(values.save_dataset),
        device=str(values.device),
    )
