"""多形状 × 四阶段 campaign 的纯函数底座：**只给形状 + 尺寸**，其余全部派生并标注来源。

为什么需要它
------------
``report/shape_trials/multishape_scale/multishape_scale_live_20261028.md`` §3.2 那次
实验的 16 段 / 112 块是把 4 个形状 × 4 个阶段逐个 lane 手敲 PowerShell 跑出来的。
手敲的 lane 有两个可复现的失败模式（都不是"人粗心"，是结构性的）：

1. **参数漂移**：同一次 campaign 里某条 lane 漏了 ``--replay-exposures-auto``，或
   把 ``--window-span`` 留成 CLI 默认 820 px（1200 px 目标需要 900 px 半跨），
   结果该形状的曝光/采样窗与其它形状不同，却和它们出现在同一张对比表里。
2. **来源丢失**：事后没人能说清某个数字是实测的、派生的、CLI 默认的，还是手填的。
   报告里 ``段首 seed CV`` 与 ``段末 final CV`` 能做配对检验，靠的正是"同一份参数"。

所以本模块把"跑一次 campaign 需要哪些参数"变成**数据**：每个参数都是一个
:class:`Setting`，带 :class:`FieldSource` 来源标签与一句中文说明（说明里写明它替身的
那个 CLI 选项或报告章节）。GUI 只需传形状与尺寸，剩下的在这里派生；人改过的值一律
标 ``MANUAL`` 并在说明里记下被替换掉的派生值，于是"这条 lane 与别的 lane 有什么不同"
是**可读的**，不需要事后考古。

本模块的边界（硬约束）
----------------------
- **不 import streamlit / torch / 任何 driver**：GUI 计划页要在无设备的机器上画表，
  导入期碰硬件包会让整个页面在没接设备时直接挂掉。
- **不 import runner**：runner import 期就绑定 ``click`` 与
  ``tools.slm.transfer.{capture,fit,validate,...}``。默认值因此抄成一张带注释的表
  :data:`RUNNER_CLI_DEFAULTS`，由
  ``tests/ao_shaping/tools/slm/transfer/test_campaign_settings.py`` 逐项与
  ``transfer_shapes_runner.run.params`` 比对 —— 改 runner 默认值而漏改这里会直接红。
- **不重算物理**：窗口档位、阶段配方分别来自
  ``live_20261028.md:41``（§5.2 尺寸窗口 420/620/900）与
  ``tools.slm.transfer.stages.STAGE_RECIPES``；本模块只做查表与配对，不另立口径。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ao_shaping.tools.slm.transfer.preflight import MIN_SIDE_PX, WINDOW_MARGIN_PX
from ao_shaping.tools.slm.transfer.stages import STAGE_RECIPES
from ao_shaping.utils.image.target import TRANSFER_SHAPE_CHOICES

__all__ = [
    "CAMPAIGN_REPLAY_EXPOSURES_AUTO",
    "DISK_GB_PER_STAGE",
    "DISK_NOTICE_GB",
    "ESTIMATE_SPREAD",
    "KNOWN_RUNNER_KWARGS",
    "MINUTES_PER_BLOCK",
    "RUNNER_CLI_DEFAULTS",
    "SECONDS_PER_BLOCK",
    "SIZE_WINDOWS",
    "CampaignSettings",
    "FieldSource",
    "MatrixRow",
    "Setting",
    "campaign_matrix",
    "derive_settings",
    "estimate_disk_gb",
    "estimate_seconds",
    "matrix_totals",
    "window_for_side",
]

#: 目标边长（原生 CCD px）→ 采样窗半跨（``--window-span``，面板 px）。
#:
#: 键是 campaign 实测过的三档尺寸，值是规划报告给每档定的半跨：报告
#: ``multishape_scale_live_20261028.md:41`` 记「窗口半跨 1200→900、600→620、
#: 200→420」，§5.2（同文件 ``:233``）记「窗口 420 / 620 / 900」。窗口必须比目标
#: 与零级圆盘都宽：200 px 方框与零级圆盘重叠 5.61 %（实测），这是**指标口径**问题，
#: 不是靠缩窗解决的 —— 所以 200 档的 420 px 反而是三档里余量最小的。
SIZE_WINDOWS: dict[float, float] = {200.0: 420.0, 600.0: 620.0, 1200.0: 900.0}

#: 单块（一次 A1(+B) 采集）的墙钟基线，分钟。原定义在
#: ``scripts/slm/plan_multishape_campaign.py``（注释原文：「measured: ~4 captures x
#: ~6 s with settle」），现以本模块为单一事实源，脚本改为 import 本常量。
#: 交叉校验：112 块 × 24 s = 2688 s，而实测四形状整形合计 2675.0 s
#: （``multishape_scale_wallclock_20261028.md:328``），偏差 0.5 %。
MINUTES_PER_BLOCK: float = 0.4

#: 每块的秒数基线（:data:`MINUTES_PER_BLOCK` 的秒表示）。
SECONDS_PER_BLOCK: float = MINUTES_PER_BLOCK * 60.0

#: 估算区间的半宽（相对值）。±40 % 不是拍出来的：wallclock 报告实测
#: ``mono12_dense`` 四块 99.7 / 140.8 / 131.8 / 125.8 s（``:128``，即 25–35 s/块），
#: 而 ``transport_cv`` 四块合计 40.0 s（``:138``，10 s/块）—— 30–40 s 的跨块散布
#: 加上阶段间差异（8 bit 段只采 A1、12 bit 段采 A1+B）就落在 ±40 % 内。
ESTIMATE_SPREAD: float = 0.4

#: 每形状每阶段的落盘体积（GB，10^9 字节），来自 wallclock 报告 §6.3 的
#: 「修复后写出字节」表（``multishape_scale_wallclock_20261028.md:225-228``）：
#: ``mono12_cv 21.10 → 1.18``、``mono12_continue 26.69 → 1.92``、
#: ``mono12_dense 5.31 → 0.33``、``transport_cv 3.00 → 0.25``。四形状合计 14.7 GB
#: （``:229``「全 campaign 写出字节：224 GB → 14.7 GB」）。
#:
#: 这是**一阶静态基线**：体积主要由 ``iterations.pkl`` 的逐块写入决定（相位数组固定
#: 640×640 float32 = 1.56 MB/条，``:161``），不随目标尺寸缩放；采集帧随采样窗增长
#: 的那部分未计入，故实际值只会偏大。
DISK_GB_PER_STAGE: dict[str, float] = {
    "transport_cv": 0.25,
    "mono12_cv": 1.18,
    "mono12_continue": 1.92,
    "mono12_dense": 0.33,
}

#: 磁盘提示阈值（GB）。落在实测 14.7 GB 的同量级；超阈值只在 ``warnings`` 里提示先看
#: 余量 —— wallclock 报告记当时机器余量 64.3 GB（``:548``），25 条 lane 的原始帧存不下
#: 未压缩版本，这正是「不压缩能省 1.13 s 但没改压缩级别」的原因。
DISK_NOTICE_GB: float = 10.0

#: 本 campaign 是否用自动重放曝光（``--replay-exposures-auto``）。
#:
#: 自动是**必须**的：固定 ladder 只在一个 lane 内成立 —— 200 px 的核在 125 ms 就满
#: 量程，而 1200 px 在 225 ms 仍可用（``--replay-exposures-auto`` 的 help 原文，
#: ``runners/slm/transfer_shapes_runner.py``）。CLI 默认是关闭（``RUNNER_CLI_DEFAULTS``
#: 里记 False），所以本 campaign 的这条是**派生值**而不是 CLI 默认值。
CAMPAIGN_REPLAY_EXPOSURES_AUTO: bool = True

#: runner 的 click 默认值，逐项照抄 ``runners/slm/transfer_shapes_runner.py`` 里对应
#: ``@click.option`` 的 ``default=``。**不在本模块 import runner**（理由见模块 docstring），
#: 改 runner 默认值必须同步改这里，否则
#: ``test_campaign_settings.py::test_default_table_matches_runner_click_defaults`` 直接红。
#:
#: ``window_span`` 也列在这里（CLI 默认 820.0，1200 px 目标放得下但比规划报告的 900 窄），
#: 但 :func:`derive_settings` 会按尺寸档位**派生**它并覆盖该默认值 —— 保留这一项是为了
#: "CLI 默认值有唯一表格"这条不出现例外。
RUNNER_CLI_DEFAULTS: dict[str, Any] = {
    "period": 128.0,  # --period
    "window_span": 820.0,  # --window-span（半跨，validate.recentre_window(half_span_px=…)）
    "zero_radius": 60.0,  # --zero-radius
    "capture_grid": 384,  # --capture-grid
    "fit_iterations": 150,  # --fit-iterations
    "control_grid": 640,  # --control-grid
    "fft_size": 2048,  # --fft-size
    "frames": 3,  # --frames
    "replay_exposures_auto": False,  # --replay-exposures-auto
    "slm_number": 1,  # --slm-number
    "camera": "daheng",  # --camera
}

#: 每个 DEFAULT 来源参数的中文说明（说明里点名它替身的 CLI 选项）。
_DEFAULT_NOTES: dict[str, str] = {
    "period": "CLI 默认（--period）：载波周期 panel px。只在 64 / 128 上标定过"
              "（preflight.CALIBRATED_PERIODS_PX），其它值是按「位移 ∝ 1/period」外推的；"
              "低于 preflight.MIN_PERIOD_PX=16 会先把目标整体推出采样窗",
    "zero_radius": "CLI 默认（--zero-radius）：零级剔除圆盘半径。剔除只改指标不改物理"
                   "（live_20261028.md §3.2 表注），所以半径必须与报告那个 5.61 % 同口径",
    "capture_grid": "CLI 默认（--capture-grid）：模型探针的 SLM 采样网格",
    "fit_iterations": "CLI 默认（--fit-iterations）：物理预测模型的拟合迭代数",
    "control_grid": "CLI 默认（--control-grid）：导出到控制分辨率用的网格（几何种子的相位面）",
    "fft_size": "CLI 默认（--fft-size）：前向传播的 FFT 边长",
    "frames": "CLI 默认（--frames）：每块每帧位深下的连拍数。多帧均值是 CV 的可重复性来源",
    "slm_number": "CLI 默认（--slm-number）：本 2f 实验系统只有 #1 号 SLM",
    "camera": "CLI 默认（--camera）：Daheng MER2-507，2592×1944、2.2 µm 像元",
}

#: 不是 runner 旗标的设置：``exposure_ms`` 由实验系统曝光探针逐块定、``zero_xy``
#: 由 fit 之后的 ``fit_stage.spot(reference)`` 实测 —— 两者都不进 ``kwargs()``，
#: 只用于开跑前的几何检查与显示。
NON_RUNNER_SETTINGS: tuple[str, ...] = ("exposure_ms", "zero_xy")

#: ``kwargs()`` 允许出现的键（= runner 的 click 参数名 + 两个位置输入）。
#: ``overrides`` 里出现表外的名字时只报警告、不静默丢弃：让拼写错误在
#: click 报「未知参数」之前就被看见，而不是悄悄少跑一个开关。
KNOWN_RUNNER_KWARGS: frozenset[str] = frozenset(
    {*RUNNER_CLI_DEFAULTS, "replay_exposures", "shapes", "side", "initial_phase", "gs_seed"}
)


class FieldSource(StrEnum):
    """一个参数值的来源。GUI 用它决定表格里那一列的灰/黑/高亮。"""

    MEASURED = "measured"  # 本会话实测（如零级位置、曝光）
    DERIVED = "derived"  # 由「形状 + 尺寸」或报告口径推出来（如窗口半跨）
    DEFAULT = "default"  # runner 的 CLI 默认值（只读，改了就是 MANUAL）
    MANUAL = "manual"  # 人改过；说明里必须记下被替换掉的值


@dataclass(frozen=True)
class Setting:
    """一个参数 + 它从哪来。

    Attributes:
        name: runner 的关键字参数名（例外见 :data:`NON_RUNNER_SETTINGS`）。
        value: 实际要用的值。
        source: :class:`FieldSource`。
        note: 一句中文说明，写清"为什么是这个值"，并点名替身的 CLI 选项/报告章节。
    """

    name: str
    value: Any
    source: FieldSource
    note: str


@dataclass(frozen=True)
class CampaignSettings:
    """一次 campaign 的完整参数集：形状 + 尺寸为输入，其余为带来源的派生结果。

    Attributes:
        shapes: 目标形状（``TRANSFER_SHAPE_CHOICES`` 的子集，已去序可见、无重复）。
        side: 目标边长（原生 CCD px）。
        settings: 参数名 → :class:`Setting`。
        warnings: 非阻断的中文提示（尺寸不在档位上、零级未测、落盘量偏大……）。
    """

    shapes: tuple[str, ...]
    side: float
    settings: Mapping[str, Setting]
    warnings: tuple[str, ...]

    def kwargs(self) -> dict[str, Any]:
        """直接喂给 ``slm-transfer-shapes`` 的关键字参数（不含 ``--out`` / ``--execute``）。

        非 runner 旗标（:data:`NON_RUNNER_SETTINGS`）已被剔除：``exposure_ms`` 由
        实验系统曝光探针定，``zero_xy`` 由 runner 在 fit 之后自己测
        （``transfer_shapes_runner.run`` 里 ``zero_xy = np.round(fit_stage.spot(reference))``），
        两者硬塞进 CLI 只会在 click 那里报未知参数。
        """
        return {
            name: setting.value
            for name, setting in self.settings.items()
            if name not in NON_RUNNER_SETTINGS
        }

    def blocking(self) -> tuple[str, ...]:
        """红停项的中文原因（空元组 = 可以开跑）。

        只判两条**结构性**的错（参数本身能不能跑），不判"结果好不好"：
        目标放不进采样窗、目标小于采样网格下限。零级重叠**不在这里**：200 px 档
        实测重叠 5.61 % 仍然跑完了（live_20261028.md:23），它是读数口径问题，
        由 :func:`ao_shaping.tools.slm.transfer.preflight.check_zero_overlap` 报给操作员。
        """
        problems: list[str] = []
        span = float(self.settings["window_span"].value)
        needed = self.side / 2.0 + WINDOW_MARGIN_PX
        if span < needed:
            problems.append(
                f"目标 {self.side:g} px 需要至少 {needed:g} px 的采样窗半跨"
                f"（半边长 + {WINDOW_MARGIN_PX:g} px 余量），当前 --window-span {span:g} px 放不下："
                "validate.recentre_window 会按这个半跨裁窗，目标会被裁掉而不是报错"
            )
        if self.side < MIN_SIDE_PX:
            problems.append(
                f"目标 {self.side:g} px 小于采样网格下限 {MIN_SIDE_PX:g} px"
                "（preflight.MIN_SIDE_PX）：2 px 步长上取不到方框内部核心，核心 CV 会退化成噪声"
            )
        return tuple(problems)


@dataclass(frozen=True)
class MatrixRow:
    """campaign 矩阵的一行 = 一个形状的一个阶段。

    Attributes:
        shape: 形状名。
        stage: 阶段名（``STAGE_RECIPES`` 的 ``name``）。
        blocks: 该阶段块数（一块 = 一次 A1(+B) 采集）。
        bit_depth: 该阶段的相机位深。
        updates: 每块的优化迭代数。
    """

    shape: str
    stage: str
    blocks: int
    bit_depth: int
    updates: int


def window_for_side(side: float) -> tuple[float, str]:
    """目标边长 → 采样窗半跨（``--window-span``），并说明命中哪一档。

    取**最近的档位**而不是"放得下的最小档"：窗口偏大只多花采集时间，偏小会把目标裁掉
    （``validate.recentre_window`` 按半跨硬裁），代价不对称。档位表 :data:`SIZE_WINDOWS`
    来自 ``live_20261028.md:41`` 的「窗口半跨 1200→900、600→620、200→420」。

    Args:
        side: 目标边长（原生 CCD px）。

    Returns:
        ``(span, note)``：半跨（px）与中文说明（命中档位 / 就近取哪一档、为什么）。

    Raises:
        ValueError: ``side`` 不是正的有限数。
    """
    side = _positive_side(side)
    band = min(SIZE_WINDOWS, key=lambda key: (abs(key - side), -key))
    span = SIZE_WINDOWS[band]
    if abs(band - side) < 1e-9:
        note = (f"尺寸 {side:g} px 命中 {band:g} px 档：采样窗半跨 {span:g} px"
                "（§5.2 尺寸窗口 420/620/900）")
    else:
        direction = "偏小" if band < side else "偏大"
        note = (f"尺寸 {side:g} px 不在 200/600/1200 档位上，按最近档 {band:g} px"
                f"（{direction}）取采样窗半跨 {span:g} px（§5.2 尺寸窗口 420/620/900）；"
                "就近取而不是取最小可容纳档：窗口偏大只多花采集时间，偏小会直接把目标裁掉")
    return span, note


def derive_settings(
    shapes: Iterable[str],
    side: float,
    *,
    overrides: Mapping[str, Any] | None = None,
    zero_xy: tuple[float, float] | None = None,
    exposure_ms: float | None = None,
    capture_grid: int | None = None,
    control_grid: int | None = None,
    fft_size: int | None = None,
    frames: int | None = None,
    replay_exposures: Sequence[float] | None = None,
    initial_phase: str = "auto",
    window_span: float | None = None,
    period: float | None = None,
    zero_radius: float | None = None,
) -> CampaignSettings:
    """只给**形状 + 尺寸**，派生出一整份带来源标签的 campaign 参数。

    每个可选关键字的位置就是"谁能覆盖谁"：

    1. **实测输入**（``zero_xy`` / ``exposure_ms``）→ ``MEASURED``；
    2. **按尺寸派生**（``window_span``）→ ``DERIVED``；
    3. **runner 的 CLI 默认值** → ``DEFAULT``；
    4. **函数参数显式给了值**（``capture_grid`` … ``replay_exposures`` …）→ ``MANUAL``；
    5. ``overrides`` 里的任何项最后生效，一律 ``MANUAL`` 并在说明里记下被替换的
       「来源 + 值」。

    Args:
        shapes: 目标形状，必须是 :data:`TRANSFER_SHAPE_CHOICES` 的非空、无重复子集。
        side: 目标边长（原生 CCD px），须为正的有限数。
        overrides: 参数名 → 值的逃生口（GUI 的"高级"区）。键可以是任何 runner 关键字参数。
        zero_xy: 实测零级位置（sensor px, ``(x, y)``）；不给就只有警告，几何检查退回标称位置。
        exposure_ms: 实测曝光（ms）；不给则为 ``"auto"``（实验系统曝光探针逐块定）。
        capture_grid: 覆盖 ``--capture-grid``。
        control_grid: 覆盖 ``--control-grid``。
        fft_size: 覆盖 ``--fft-size``。
        frames: 覆盖 ``--frames``。
        replay_exposures: 显式重放曝光 ladder；给了就关掉自动模式（``replay_exposures_auto=False``）。
        initial_phase: 初相位构造；``"auto"`` 保持 runner 的 ``_seed_kind`` 规则。
        window_span: 覆盖按尺寸派生的采样窗半跨。
        period: 覆盖 ``--period``。
        zero_radius: 覆盖 ``--zero-radius``。

    Returns:
        :class:`CampaignSettings`。

    Raises:
        ValueError: 形状非法/为空/重复，或边长非正 —— 中文消息，GUI 直接当红停原因显示。
    """
    shape_tuple = _validated_shapes(shapes)
    side_value = _positive_side(side)
    explicit = dict(overrides or {})

    settings: dict[str, Setting] = {}
    warnings: list[str] = []

    def put(name: str, value: Any, source: FieldSource, note: str) -> None:
        settings[name] = Setting(name=name, value=value, source=source, note=note)

    # ---- 实测输入 -------------------------------------------------------
    if zero_xy is None:
        warnings.append(
            "零级位置未测：几何检查只能按标称位置，运行前请做光路探针"
            "（scripts/slm/probe_optical_path_validity.py；零级与目标框的重叠面积是读数口径的一部分）"
        )
    else:
        x, y = (float(v) for v in zero_xy)
        put("zero_xy", (x, y), FieldSource.MEASURED,
            "实测零级位置（sensor px）：runner 在 fit 之后用 "
            "fit_stage.spot(reference) 自己测；这里传入只为开跑前算零级剔除区与目标框的重叠"
            "（preflight.check_zero_overlap），不是 runner 旗标")

    if exposure_ms is None:
        put("exposure_ms", "auto", FieldSource.DEFAULT,
            "auto = 由实验系统曝光探针逐块定（--replay-exposures-auto 的自动 ladder "
            "按本 lane 实测核心峰值率推导，见 transfer_shapes_runner 的同名旗标）；"
            "本 runner 没有 --exposure-ms 旗标，所以它不进 kwargs()")
    else:
        put("exposure_ms", float(exposure_ms), FieldSource.MEASURED,
            "实测固定曝光（ms）：写进每次采集的 manifest，使各块可直接比较；"
            "固定曝光只在一个 lane 内成立（200 px 核在 125 ms 满量程、1200 px 在 225 ms 仍可用）")

    # ---- 按尺寸派生 -----------------------------------------------------
    span, span_note = window_for_side(side_value)
    if window_span is not None:
        put("window_span", float(window_span), FieldSource.MANUAL,
            f"手工覆盖：派生值 {span:g} px 被替换（{span_note}）")
    else:
        put("window_span", span, FieldSource.DERIVED,
            f"{span_note}；取代 CLI 默认 {RUNNER_CLI_DEFAULTS['window_span']:g} px"
            "（--window-span 是采样窗半跨，validate.recentre_window(half_span_px=…)）")
    if abs(side_value - min(SIZE_WINDOWS, key=lambda key: abs(key - side_value))) > 1e-9:
        warnings.append(
            f"尺寸 {side_value:g} px 不在窗口档位 200/600/1200 上，已就近取 {span:g} px 半跨："
            "该尺寸没有实测的窗口/曝光组合，结果只能与同尺寸的 lane 配对比较"
        )

    if period is not None:
        put("period", float(period), FieldSource.MANUAL,
            f"手工覆盖：CLI 默认 {RUNNER_CLI_DEFAULTS['period']:g} 被替换（载波位移 ∝ 1/period）")
    else:
        put("period", RUNNER_CLI_DEFAULTS["period"], FieldSource.DEFAULT, _DEFAULT_NOTES["period"])

    if zero_radius is not None:
        put("zero_radius", float(zero_radius), FieldSource.MANUAL,
            f"手工覆盖：CLI 默认 {RUNNER_CLI_DEFAULTS['zero_radius']:g} 被替换"
            "（改半径就改了零级剔除面积，须与报告口径一起读）")
    else:
        put("zero_radius", RUNNER_CLI_DEFAULTS["zero_radius"], FieldSource.DEFAULT,
            _DEFAULT_NOTES["zero_radius"])

    # ---- runner CLI 默认值 ----------------------------------------------
    explicit_defaults: dict[str, Any] = {
        "capture_grid": capture_grid, "control_grid": control_grid,
        "fft_size": fft_size, "frames": frames,
    }
    for name in ("capture_grid", "fit_iterations", "control_grid", "fft_size", "frames",
                 "slm_number", "camera"):
        explicit_value = explicit_defaults.get(name)
        if explicit_value is not None:
            put(name, explicit_value, FieldSource.MANUAL,
                f"手工覆盖：CLI 默认 {RUNNER_CLI_DEFAULTS[name]!r} 被替换")
        else:
            put(name, RUNNER_CLI_DEFAULTS[name], FieldSource.DEFAULT, _DEFAULT_NOTES[name])

    # ---- 重放曝光 -------------------------------------------------------
    if replay_exposures is None:
        put("replay_exposures_auto", CAMPAIGN_REPLAY_EXPOSURES_AUTO, FieldSource.DERIVED,
            "本 campaign 打开自动重放曝光（--replay-exposures-auto，CLI 默认 "
            f"{RUNNER_CLI_DEFAULTS['replay_exposures_auto']!r}）：固定 ladder 只在一个 lane 内成立 —— "
            "200 px 的核在 125 ms 就满量程，1200 px 在 225 ms 仍可用，跨尺寸共用会把小尺寸读数削顶")
    else:
        ladder = tuple(float(v) for v in replay_exposures)
        put("replay_exposures", ladder, FieldSource.MANUAL,
            "手工覆盖：显式重放 ladder 替换了自动模式（自动值由本 lane 实测核心峰值率推导）")
        put("replay_exposures_auto", False, FieldSource.MANUAL,
            f"手工覆盖：自动模式已关闭（CLI 默认 {RUNNER_CLI_DEFAULTS['replay_exposures_auto']!r}）")

    # ---- 初相位 ---------------------------------------------------------
    auto_rule = "正方形用传输相位（transport，二维可分离、对正方形精确），其余形状用反投影（backprojection，形状通用）"
    if initial_phase == "auto":
        put("initial_phase", "auto", FieldSource.DEFAULT,
            f"CLI 默认（--initial-phase auto）：{auto_rule} —— runner._seed_kind 的规则。"
            "严格冷启动（random / plane，不读目标也不用物理预测模型）与解析初始化"
            "（transport / backprojection / gs，用了拟合模型）**不可并入同一张对比表**："
            "后者赢了不能证明「闭环从零开始也能工作」（plan_multishape_campaign.py 开头）")
    else:
        put("initial_phase", initial_phase, FieldSource.MANUAL,
            f"手工覆盖：CLI 默认 auto（{auto_rule}）被替换为 {initial_phase!r}")

    # ---- campaign 协议派生 ----------------------------------------------
    put("gs_seed", False, FieldSource.DERIVED,
        "本次 campaign 是「无 GS」方案：--no-gs-seed 关掉对比择优里的 GS 候选"
        "（--gs-seed 的 help 原文：「无 GS 冷启动方案须用 --no-gs-seed」；"
        "live_20261028.md §5.2 的 lane 即带 --no-gs-seed）")

    put("shapes", shape_tuple, FieldSource.MANUAL,
        "输入：目标形状（GUI 的主输入之一）。顺序即跑批顺序，--order shape 时一形状跑完四段")
    put("side", side_value, FieldSource.MANUAL,
        "输入：目标边长（原生 CCD px）。它同时决定采样窗档位、目标框几何与零级重叠面积")

    # ---- 人的覆盖最后生效 ------------------------------------------------
    for name, value in explicit.items():
        if name not in KNOWN_RUNNER_KWARGS:
            warnings.append(
                f"未知参数 {name!r}：本页没有它的派生值，它会原样写进 kwargs() —— "
                "runner 会以未知参数报错，请确认拼写（可能是想改 "
                f"{sorted(KNOWN_RUNNER_KWARGS)[:3]}… 之一）"
            )
        previous = settings.get(name)
        if previous is None:
            put(name, value, FieldSource.MANUAL, "手工覆盖：本页没有派生值，直接采用")
        else:
            put(name, value, FieldSource.MANUAL,
                f"手工覆盖：{previous.source} 值 {previous.value!r} 被替换")

    rows = campaign_matrix(shape_tuple)
    disk = estimate_disk_gb(rows)
    if disk > DISK_NOTICE_GB:
        seconds = estimate_seconds(rows)[1]
        warnings.append(
            f"本 campaign 预计落盘 ≈{disk:.1f} GB、整形墙钟 ≈{_human_duration(seconds)}："
            "开跑前先确认磁盘余量（同规模实测 14.7 GB 写出量，当时机器余量 64.3 GB）"
        )

    return CampaignSettings(shapes=shape_tuple, side=side_value,
                            settings=settings, warnings=tuple(warnings))


def campaign_matrix(shapes: Iterable[str]) -> tuple[MatrixRow, ...]:
    """形状 × 阶段 的 campaign 矩阵（4 形状 → 16 行 / 112 块）。

    块数直接取 :data:`STAGE_RECIPES`（4 / 8 / 12 / 4 = 每形状 28 块），与
    ``multishape_scale_live_20261028.md:64``（§3.2）「四形状合计 112 块」一致。

    Raises:
        ValueError: 形状非法/为空/重复。
    """
    shape_tuple = _validated_shapes(shapes)
    return tuple(
        MatrixRow(shape=shape, stage=recipe.name, blocks=recipe.blocks,
                  bit_depth=recipe.bit_depth, updates=recipe.updates)
        for shape in shape_tuple
        for recipe in STAGE_RECIPES
    )


def matrix_totals(rows: Sequence[MatrixRow]) -> dict[str, int]:
    """矩阵合计：段数 / 块数 / 迭代数 / 形状数（4 形状 → ``segments=16, blocks=112``）。"""
    return {
        "segments": len(rows),
        "blocks": sum(row.blocks for row in rows),
        "updates": sum(row.updates for row in rows),
        "shapes": len({row.shape for row in rows}),
    }


def estimate_seconds(rows: Sequence[MatrixRow]) -> tuple[float, float, float]:
    """整形墙钟估算 ``(乐观, 期望, 悲观)``，秒。

    期望值 = 总块数 × :data:`SECONDS_PER_BLOCK`（不做逐阶段加权：所有阶段的"一块"
    都是同一套采集+优化+落盘流程，阶段间差异正是 :data:`ESTIMATE_SPREAD` 要吸收的）。
    交叉校验：112 块 → 2688 s，实测四形状整形合计 2675.0 s
    （``multishape_scale_wallclock_20261028.md:328``）。

    口径：**只算整形块**。capture / fit / 几何重导出 / 重放不在内 —— 它们与块数无关，
    复用已有模型时整段跳过（同一份报告记 capture 173.2 + fit 31.2 + capture_geometry
    132.3 s，``:252``）。
    """
    blocks = sum(row.blocks for row in rows)
    expected = blocks * SECONDS_PER_BLOCK
    return (
        round(expected * (1.0 - ESTIMATE_SPREAD), 3),
        round(expected, 3),
        round(expected * (1.0 + ESTIMATE_SPREAD), 3),
    )


def estimate_disk_gb(rows: Sequence[MatrixRow]) -> float:
    """落盘体积估算（GB），按阶段查 :data:`DISK_GB_PER_STAGE` 再乘形状数。

    只按矩阵里的阶段求和，不重算字节：体积主导项是每次优化迭代整体重写
    ``iterations.pkl``（修复后改为每块一次），与目标尺寸无关。
    """
    return round(sum(DISK_GB_PER_STAGE.get(row.stage, 0.0) for row in rows), 3)


def _human_duration(seconds: float) -> str:
    """秒 → 「x 小时 y 分钟」的中文短句（只用于提示文本）。"""
    minutes = seconds / 60.0
    if minutes < 1.0:
        return f"{seconds:.0f} 秒"
    if minutes < 60.0:
        return f"{minutes:.0f} 分钟"
    return f"{minutes / 60.0:.1f} 小时"


def _validated_shapes(shapes: Iterable[str]) -> tuple[str, ...]:
    """校验形状：非空、都在词表内、无重复。"""
    if isinstance(shapes, str):  # 常见误用：把 "square" 当成形状列表传进来
        raise ValueError(
            f"shapes 需要形状的序列，收到单个字符串 {shapes!r}；"
            f"单个形状请写 ({shapes!r},)，可用形状: {list(TRANSFER_SHAPE_CHOICES)}"
        )
    try:
        items = tuple(shapes)
    except TypeError:
        raise ValueError("shapes 必须是一个可迭代的形状序列") from None
    if not items:
        raise ValueError("至少要选一个目标形状（可用形状: "
                         f"{list(TRANSFER_SHAPE_CHOICES)}）")
    unknown = [item for item in items if item not in TRANSFER_SHAPE_CHOICES]
    if unknown:
        raise ValueError(
            f"未知形状 {unknown}；可用形状: {list(TRANSFER_SHAPE_CHOICES)}"
        )
    duplicated = sorted({item for item in items if items.count(item) > 1})
    if duplicated:
        raise ValueError(
            f"形状重复 {duplicated}：同一形状跑两遍会把同一个 lane 写进同一目录"
            "（validate.run 对已存在的阶段目录抛 FileExistsError）"
        )
    return items


def _positive_side(side: float) -> float:
    """校验边长：正的有限数（px）。"""
    try:
        value = float(side)
    except (TypeError, ValueError):
        raise ValueError(f"目标边长必须是数字（px），收到 {side!r}") from None
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"目标边长必须是正的有限数（px），收到 {side!r}")
    return value
