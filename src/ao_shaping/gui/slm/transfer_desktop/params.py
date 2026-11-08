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
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ao_shaping.utils.image.target.metrics import POLYGON_SHAPE_CHOICES
from ao_shaping.utils.io.timestamp import format_ts
from ao_shaping.tools.slm.transfer.preflight import (
    MIN_SIDE_PX as _MIN_SIDE_PX,
    WINDOW_MARGIN_PX as _WINDOW_MARGIN_PX,
)

if TYPE_CHECKING:
    from ao_shaping.tools.slm.transfer.workflow import TransferSession

__all__ = [
    "ACCEPTANCE_LOWER_DEFAULT",
    "ACCEPTANCE_UPPER_DEFAULT",
    "ASPECT_RATIO_SESSION_FIELD",
    "CAMERA_CHOICES",
    "SHAPE_CHOICES",
    "SHAPE_LABELS",
    "STAGE_LABELS",
    "TILT_SESSION_FIELD",
    "TransferFormValues",
    "build_session",
    "effective_window_span",
    "plan_session_dir",
    "validate_form",
]

#: 物理 lane（``validate.run`` / ``transfer_shapes_runner``）消费倾斜相位开关时要
#: 找的会话字段名。``TransferSession`` 当前没有这个字段（workflow.py 不在本模块的
#: 写域），:func:`build_session` 会把它以运行时属性挂到返回值上，契约字段名就是这串
#: 常量。物理 lane 读 ``session.tilt``（等价 ``args.tilt``）决定要不要把目标用倾斜相位
#: 推离零级；关闭（默认）时目标留在零级、倾斜相位周期不适用。
TILT_SESSION_FIELD = "tilt"

#: 物理 lane 消费矩形「长/短比」要找的会话字段名。语义：``side`` 是矩形的**短边**，
#: ``aspect_ratio`` 是**长/短比**（长边 = 短边 × 长/短比）。非矩形形状固定 ``1.0``。
#: 与 ``TILT_SESSION_FIELD`` 一样以运行时属性挂在 :func:`build_session` 的返回值上。
ASPECT_RATIO_SESSION_FIELD = "aspect_ratio"

#: 未启用倾斜相位（tilt=False）时倾斜相位周期 / 位移不适用，需要屏蔽掉的 preflight 结论码。
#: 没有倾斜相位把目标推出零级，这些判据（周期可表示性、位移是否溢出采样窗）就无从谈起。
_TILT_DEPENDENT_CHECK_CODES: frozenset[str] = frozenset(
    {
        "period_invalid",
        "period_below_nyquist",
        "period_extrapolated",
        "side_offset_overflow",
    }
)

#: 矩形长/短比的合理上限（越大越极端，超过它在采样窗里既难容纳也失真）。
_RECT_ASPECT_MAX = 10.0

#: 验收阈值循环：相关性（correlation）下界的算法建议值，也是页面控件的默认值。
#: 与 :class:`TransferFormValues` 里 ``acceptance_lower`` 的默认一致（镜像
#: ``TransferSession.acceptance_lower``）；页面「算法建议」提示读取这里。
ACCEPTANCE_LOWER_DEFAULT = 0.85  # correlation 下界（算法建议）

#: 验收阈值循环：relative_l2 上界的算法建议值，页面控件的默认值。
#: 与 ``acceptance_upper`` 默认一致（镜像 ``TransferSession.acceptance_upper``）。
ACCEPTANCE_UPPER_DEFAULT = 0.25  # relative_l2 上界（算法建议）

#: 相机后端选项，与 ``bench_settings.CAM_TYPE_OPTIONS`` 同序。迁移学习只对大恒
#: 相机做过实验验证，其余后端由 :func:`validate_form` 在启动前拒绝。
CAMERA_CHOICES: tuple[str, ...] = ("daheng", "miicam", "sim")

#: 目标形状选项（顺序即下拉/勾选的显示顺序）。镜像
#: ``utils.image.target.metrics.POLYGON_SHAPE_CHOICES``。
SHAPE_CHOICES: tuple[str, ...] = POLYGON_SHAPE_CHOICES

#: 形状 → 中文名。镜像 ``transfer_candidates.TRANSFER_SHAPE_LABELS``（该模块
#: import Streamlit，桌面页不能依赖）。
SHAPE_LABELS: dict[str, str] = {
    "square": "正方形",
    "rectangle": "长方形",
    "triangle": "三角形",
    "star": "五角星",
}

#: ``run_session`` 发出的阶段名 → 操作员读得懂的标签。values 与
#: ``transfer_progress.STAGE_LABELS`` 的**键**一一对应（阶段名不能漂移），个别**值**
#: 这里写得更直白（例如阶段名 ``CCD_model_blocks`` →「CCD 配对块」）。测试钉住键集。
STAGE_LABELS: dict[str, str] = {
    "capture": "当前光路采集（探针 + 逐帧自动曝光）",
    "reuse_capture": "复用已有采集",
    "optical_probe": "光路探测（复用判定）",
    "nominal_initialization": "初始模型准备（冷启动，先用几何设计初值起手，未实测标定）",
    "fit": "物理预测模型迁移 / 拟合",
    "CCD_model_blocks": "CCD 实测对照块（模型预测 vs 实测）",
    "exposure_response_capture": "曝光阶梯诊断",
    "exposure_response_analysis": "曝光响应分析",
    "blur_calibration": "有效展宽 / 辐射尺度校准",
    "CCD_anchored_blocks": "CCD 实测校正优化块（按实测迭代，验收：A–B–B–A 或仅模型损失）",
    "GS_and_differentiable_synthesis": "GS + 可微相位合成（离线）",
    "transferred_phase_differentiable_synthesis": (
        "历史接受相位迁移 + 可微合成（离线）"
    ),
    "raw_data_audit": "原始数据审计",
    "summary": "指标摘要",
    "direction_check": "方向核对",
}

#: 验收阈值循环第 N 轮（N ≥ 1）的阶段名前缀，由 ``run_session`` 拼出：
#: ``acceptance_round_{N}_{base}``。第 0 轮不带前缀。
_ROUND_PREFIX_RE = re.compile(r"^acceptance_round_(\d+)_(.+)$")


def stage_label(name: str, fallback: str | None = None) -> str:
    """阶段名 → 操作员标签；循环第 N 轮的阶段显示为「第 N 轮 · 基础标签」。

    未知名称回落到 ``fallback``（缺省则原样返回名称）。
    """
    match = _ROUND_PREFIX_RE.match(name)
    if match:
        base = STAGE_LABELS.get(match.group(2), match.group(2))
        return f"第 {match.group(1)} 轮 · {base}"
    return STAGE_LABELS.get(name, fallback if fallback is not None else name)


@dataclass(frozen=True)
class TransferFormValues:
    """桌面页表单的纯数据快照（与 Streamlit 的 ``values`` 字典一一对应）。

    字段顺序即表单顺序：目标 → 预算 → 相机 / SLM → 高级口径 → 历史产物路径。
    所有字段都带有限默认值（镜像 ``TRANSFER_RECIPE`` 与 Streamlit 控件的
    ``value=``），因此空表单也能直接构造。

    Attributes:
        shapes: 目标形状（``SHAPE_CHOICES`` 的子集）。
        side: 目标尺寸 (CCD px)。矩形时为**短边**（长边 = 短边 × ``aspect_ratio``）。
        tilt: 是否添加倾斜相位。``False``（默认）时目标留在零级 (0 级)，
            倾斜相位周期 / 位移不适用；``True`` 时用倾斜相位把目标推离零级。
        aspect_ratio: 矩形**长/短比**（仅 ``shapes`` 含 ``rectangle`` 时有意义，
            见 :data:`_RECT_ASPECT_MAX`）；非矩形形状固定 ``1.0``。
        period: 倾斜相位周期 (panel px)。
        count: 分组随机探针数（必须是 6 的倍数）。
        iterations: 模型参数拟合迭代数（最大预算；早停可提前终止）。
        patience: 早停 patience（连续无改进的迭代数）。
        min_delta: 早停最小改进幅度。
        blocks: 按 CCD 实测校正的短块数。
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
        acceptance_enabled: 是否开启验收阈值循环。``False``（默认）保持当前单轮
            行为完全不变；``True`` 时 ``run_session`` 拟合后在验收行上比对 CCD 与
            模型预测，不达标则把验证数据并入训练并重训，直到达标、用户点「停止」
            或达到 :data:`acceptance_max_rounds` 上限。
        acceptance_metric: 循环用的验收口径（``quality`` 结合相关性与 relative_l2）。
        acceptance_lower: 相关性（correlation）下界，算法建议之（见
            :data:`ACCEPTANCE_LOWER_DEFAULT`）。区间 (0, 1]。
        acceptance_upper: relative_l2 上界，算法建议值（见
            :data:`ACCEPTANCE_UPPER_DEFAULT`）。须 > 0。
        acceptance_max_rounds: 循环的安全上限（重训轮数），到限即优雅停止。
    """

    shapes: tuple[str, ...] = ("square",)
    side: float = 80.0
    tilt: bool = False
    aspect_ratio: float = 2.0
    period: float = 64.0
    count: int = 24
    iterations: int = 200
    patience: int = 8
    min_delta: float = 1e-4
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
    acceptance_enabled: bool = False
    acceptance_metric: str = "quality"
    acceptance_lower: float = 0.85
    acceptance_upper: float = 0.25
    acceptance_max_rounds: int = 8
    #: 光路复用：True = 先探测光路是否与最近一份 latest 模型一致，一致则跳过
    #: 采集 / 拟合，直接把那次 CCD 对照重做一遍；False = 当前全流程重采集。
    reuse_latest: bool = False


def effective_window_span(
    side: float,
    window_span: float = 0.0,
    aspect_ratio: float = 1.0,
) -> float:
    """采样窗**半跨**（px）：显式值优先，否则由目标边长派生。

    为什么必须派生
    --------------
    采样窗是 ``validate.run`` 里 ``recentre_window(zero_xy, half_span_px=...)``
    的半径：目标斑整个落在窗内，两轴各再留 :data:`WINDOW_MARGIN_PX` 的边，
    窗才既框住目标又不压窗边。管线默认半跨 220 px 只够 424 px 的目标，而实测
    光路上 ``side=900`` 是在线实现过的 —— 上限的唯一物理依据是 0 级到帧边的距离
    （页面「读取零级」量出来），不是这个默认值。

    于是：

    * ``window_span > 0``（显式，例如命令行已验证过的 930）→ 原样返回；
    * 否则 ``max(220, extent/2 + 8)``：采样窗必须放下整个目标 + 两侧留白。
      ``extent`` 大到接近帧边时，派生出的半跨恰好等于 0 级到帧边的距离，
      窗刚好不越界 —— 这正是「目标边长上限」的物理含义。

    小目标行为不变：``side <= 424`` 时结果仍是 220.0，与管线默认值逐位相同。

    Args:
        side: 目标边长（CCD px）。
        window_span: 显式半跨（px）；``0`` = 自动派生。
        aspect_ratio: 矩形长/短比。采样窗的
            ``recentre_window`` 裁的是方形半跨（两轴同半径），所以派生时必须按
            目标在轴上占到的**最大跨度** ``side * aspect_ratio`` 来算 —— 只按短边
            派生会让长边溢出窗（短 130 × 比 2 → 长边 260 > 窗 220）。非矩形传
            ``1.0``，此时 ``extent == side``，行为不变。

    Returns:
        采样窗半跨（px），**不是全宽**。
    """
    from ao_shaping.tools.slm.transfer.preflight import (
        HALF_SPAN_PX,
        WINDOW_MARGIN_PX,
    )

    if float(window_span) > 0:
        return float(window_span)
    extent = float(side) * max(float(aspect_ratio), 1.0)
    return max(HALF_SPAN_PX, extent / 2.0 + WINDOW_MARGIN_PX)


def validate_form(values: TransferFormValues) -> tuple[list[str], list[str]]:
    """按 Streamlit 页的顺序做运行前校验。

    顺序固定（与 ``TransferMethod.validate`` 一致）：torch 未安装 → 相机类型
    不是大恒 → 没选形状 → ``preflight.check_parameters``。前三条任何一条不通过
    都直接返回：缺 torch 时讨论倾斜相位没有意义，也不会把一条"周期性"提示混进
    真正的阻塞原因里。开启验收阈值循环时，另对相关性下界 / relative_l2 上界做
    一次**提醒级**边界检查（不阻塞）。

    ``values.tilt`` 关闭（默认）时目标留在零级 (0 级)，倾斜相位周期 / 位移不适用，
    因此屏蔽掉那些倾斜相位专属的 preflight 判据（可表示性、位移是否溢出采样窗）——
    否则关掉倾斜相位仍会收到"周期不在标定档"之类的报错。``tilt=True`` 时行为
    与旧版完全一致。矩形额外要求长 > 短 > 0、长/短都在采样窗内、长/短比在
    :data:`_RECT_ASPECT_MAX` 以内。

    Args:
        values: 表单快照。

    Returns:
        ``(errors, warnings)``。元素是可直接显示的中文文本；``preflight`` 的结论
        写成「哪里不对 + ``→`` 怎么改」两行。``errors`` 非空即禁止启动，
        ``warnings`` 只提示（例如未标定档的倾斜相位周期）。
    """
    torch_error = _torch_missing()
    if torch_error is not None:
        return [torch_error], []

    camera = str(values.camera)
    if camera not in ("daheng", "miicam"):
        return [
            f"迁移学习闭环目前只对大恒 / MiiCam 相机做过实验验证（相机类型 {camera!r}）。"
            "其他后端仅有接口支持；请在「相机类型」里选择 daheng 或 miicam"
        ], []

    if not values.shapes:
        return ["至少选择一个目标形状"], []

    tilt = bool(values.tilt)
    shapes = tuple(values.shapes or ())
    side = float(values.side)
    period = float(values.period)

    if "rectangle" in shapes:
        aspect_ratio = float(values.aspect_ratio)
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 1.0:
            return [
                f"矩形长/短比 {aspect_ratio!r} 必须大于 1（长边须严格大于短边）。"
                "\n    → 填 1–10 之间的值。"
            ], []
        if aspect_ratio > _RECT_ASPECT_MAX:
            return [
                f"矩形长/短比 {aspect_ratio:g} 超出合理上限 {_RECT_ASPECT_MAX:g}。"
                "\n    → 太大在采样窗里既难容纳也失真，请调小。"
            ], []
        window_span = effective_window_span(side, values.window_span, aspect_ratio)
        long_side = side * aspect_ratio
        # 采样窗是半跨：两轴各能容纳 ``window_span - 留白`` 的半个目标，故整边能放进
        # ``2*(window_span - margin)`` —— 与 preflight._max_side_px 同语义。只按半跨
        # 硬比会把「放得进窗」的目标误判成超窗（如短边 130×比 2，长边 260 落在
        # 半跨 220 已经派生放大的窗里）。
        side_limit = 2.0 * (window_span - _WINDOW_MARGIN_PX)
        for label, value in ("短边", side), ("长边", long_side):
            if not math.isfinite(value) or value <= 0:
                return [f"矩形{label} {value!r} 不是正数。"], []
            if value > side_limit:
                return [
                    f"矩形{label} {value:g} px 超出采样窗可容纳的 {side_limit:g} px，"
                    "放不进窗。"
                    "\n    → 请缩小边长或长/短比。"
                ], []
            if value < _MIN_SIDE_PX:
                return [
                    f"矩形{label} {value:g} px 小于最小边长 {_MIN_SIDE_PX:g} px。"
                ], []
    else:
        aspect_ratio = 1.0

    from ao_shaping.tools.slm.transfer.preflight import check_parameters

    problems = check_parameters(
        side=side,
        period=period,
        count=int(values.count),
        iterations=int(values.iterations),
        blocks=int(values.blocks),
        updates=int(values.updates),
        frames=int(values.frames),
        shapes=shapes,
        half_span_px=effective_window_span(side, values.window_span, aspect_ratio),
    )
    if not tilt:
        problems = [
            problem for problem in problems if problem.code not in _TILT_DEPENDENT_CHECK_CODES
        ]

    errors: list[str] = []
    warnings: list[str] = []
    for problem in problems:
        text = f"{problem.message}\n    → {problem.hint}"
        if problem.severity == "error":
            errors.append(text)
        else:
            warnings.append(text)
    # 采集网格自动升档：升档时给一条中性说明；目标在任何合法网格下都够不着时直接阻塞
    # （空跑一遍采集只会浪费设备时间）。升档成功时吊销原先「目标超出可达窗」的提示，避免
    # 同一目标出现互相矛盾的红黄两条。
    grid_info, grid_error = describe_capture_grid(
        side=side, shapes=shapes, aspect_ratio=aspect_ratio, tilt=tilt, period=period,
    )
    if grid_error:
        errors.append(grid_error)
    if grid_info:
        # 中性说明（不是警告）：自动升档是预期行为，只告诉操作员网格会变成多少。
        warnings.append(grid_info)
    # 验收阈值循环的边界（只在校验开启时把关, 关闭 = 当前单轮行为, 不拦不提示）。
    # 阈值是算法建议的浮动下/上界: 给越界一个提醒, 不阻塞——操作员可以按自己的
    # 判定放宽, 但提醒能防住把 correlation 拖到 0 以下这类明显笔误。
    if bool(values.acceptance_enabled):
        lower = float(values.acceptance_lower)
        upper = float(values.acceptance_upper)
        if not math.isfinite(lower) or lower <= 0.0 or lower > 1.0:
            warnings.append(
                f"相关性下界 {lower!r} 不在 (0, 1] 区间内（算法建议 {ACCEPTANCE_LOWER_DEFAULT}）。"
                "\n    → 验收循环会拿它对比模型预测的相关性，请填 0 到 1 之间的值。"
            )
        if not math.isfinite(upper) or upper <= 0.0:
            warnings.append(
                f"relative_l2 上界 {upper!r} 不是正数（算法建议 {ACCEPTANCE_UPPER_DEFAULT}）。"
                "\n    → 验收循环会拿它对比模型预测的相对 L2 误差，请填大于 0 的值。"
            )
    return errors, warnings


def default_capture_grid() -> int:
    """``TransferSession.capture_grid`` 的默认值——桌面没有网格控件，基准就是它。"""
    from dataclasses import fields

    from ao_shaping.tools.slm.transfer.workflow import TransferSession

    return int(next(f.default for f in fields(TransferSession) if f.name == "capture_grid"))


def describe_capture_grid(
    *, side: float, shapes, aspect_ratio: float, tilt: bool, period: float,
) -> tuple[str | None, str | None]:
    """目标在采集网格自动升档下的结论：``(说明文案, 阻塞文案)``，各至多一条。

    判据与会话里 :func:`decide_capture_grid` 完全同源（同一份 ``required_capture_grid``），
    所以表单与真正运行看到的网格永远一致。位移只在倾斜相位开启时计入（与 ``tilt``
    关闭时目标留在零级一致）。纯函数、不抛：任何异常都退回「不说话」。

    - 目标在某个合法网格下够得着 → 说明文案，写出升到哪一档、可达半径；
    - 任何合法网格都够不着 → 阻塞文案，写出最大可容纳边长；
    - 目标本来就放得下 → ``(None, None)``，不打扰。
    """
    try:
        from ao_shaping.algorithm.signal_processing.multishape_objective import (
            _fft_reach_px,
            capture_grid_ladder,
            carrier_offset_px,
            fft_max_side_px,
            required_capture_grid,
            target_extent_px,
        )
        from ao_shaping.utils.slm.bench_constants import TILT_SHIFT_SCALE

        shape_tuple = tuple(shapes or ("square",))
        from ao_shaping.tools.slm.transfer.preflight import MIN_PERIOD_PX

        if tilt and (not math.isfinite(float(period)) or float(period) < MIN_PERIOD_PX):
            # 周期本身已是阻塞错误：此时外推的位移没有物理意义，不再补一条升档说明。
            return None, None
        offset = carrier_offset_px(float(period), TILT_SHIFT_SCALE) if tilt else 0.0
        extent = target_extent_px(float(side), shapes=shape_tuple, aspect_ratio=float(aspect_ratio))
        if extent <= 0:
            return None, None
        ladder = capture_grid_ladder()
        widest = ladder[-1]
        needed = required_capture_grid(
            float(side), TILT_SHIFT_SCALE, shapes=shape_tuple,
            aspect_ratio=float(aspect_ratio), offset_px=offset,
        )
        if needed is None:
            limit = fft_max_side_px(widest, TILT_SHIFT_SCALE, offset_px=offset)
            return None, (
                f"目标 {extent:g} px 超出任何合法采集网格的可达窗（最宽 {widest} 也只到 "
                f"{limit:.0f} px）。\n    → 把目标边长降到 ≤ {limit:.0f} px"
                "（或长/短比），否则采集跑完也无法整形。"
            )
        if needed <= default_capture_grid():
            return None, None  # 默认网格已够得着：不升档，就不打扰
        reach = _fft_reach_px(needed, TILT_SHIFT_SCALE)
        return (
            f"采集网格自动升档：{default_capture_grid()} → {needed}"
            f"（目标 {extent:g} px 需可达半径 ≥{extent / 2:.0f} px，升档后 ±{reach:.0f} px）"
        ), None
    except Exception:  # 显示路径兜底：校验页绝不因为提示计算而崩
        return None, None


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
    session = TransferSession(
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
        patience=int(values.patience),
        min_delta=float(values.min_delta),
        period=float(values.period),
        window_span=effective_window_span(
            float(values.side),
            values.window_span,
            float(values.aspect_ratio)
            if "rectangle" in tuple(values.shapes or ())
            else 1.0,
        ),
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
        acceptance_enabled=bool(values.acceptance_enabled),
        acceptance_metric=str(values.acceptance_metric),
        acceptance_lower=float(values.acceptance_lower),
        acceptance_upper=float(values.acceptance_upper),
        acceptance_max_rounds=int(values.acceptance_max_rounds),
        reuse_latest=bool(values.reuse_latest),
        # 桌面不暴露网格控件：采集网格由目标尺寸自动定（只升不降，见 decide_capture_grid）。
        auto_capture_grid=True,
    )
    # ``TransferSession`` 没有 tilt / aspect_ratio 字段（workflow.py 不在写域），
    # 物理 lane（``validate.run``）按 :data:`TILT_SESSION_FIELD` /
    # :data:`ASPECT_RATIO_SESSION_FIELD` 这对契约字段名从会话上读，于是这里把
    # 它们以运行时属性挂上去；``frozen`` 会话直接赋值会抛，故走 object.__setattr__。
    object.__setattr__(session, TILT_SESSION_FIELD, bool(values.tilt))
    aspect_hook = 1.0
    if "rectangle" in (values.shapes or ()):
        aspect_hook = float(values.aspect_ratio)
    object.__setattr__(session, ASPECT_RATIO_SESSION_FIELD, aspect_hook)
    return session
