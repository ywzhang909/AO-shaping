"""整形任务的路由决策：把小规模探测报告映射为推荐的整形方法组合。

本模块**纯离线**：只 import 标准库，不触碰 torch / 设备驱动 / Qt，
目的是在任何无设备环境（含 CI、单测）下都能确定性工作。

依据规则来自 ``report/shape_trials/experience_review/shaping_experience_20261103.md``
§1.1「按任务选方法」+ §3 固定协议。它**只给建议**，不执行大优化；
最终执行与否由用户确认后运行。

规则优先级从高到低，命中即定：

1. ``no_go``      —— 噪声高于相位差效应（只给整改建议，不整形）。
2. ``extended_field``  —— 超视场建模/物理项研究。
3. ``refocus``    —— 光路/焦面刚变化（任一证据成立）。
4. ``cold_start`` —— 冷启动（无 checkpoint 且无 seed run）。
5. ``new_mask``   —— 新掩膜或未标定形状。
6. ``anchored_uniformity`` —— 匀化意图或冻结模型锚点偏离。
7. ``stable_small``（默认）—— 稳定小幅整形。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

# ---------------------------------------------------------------------------
# 阈值（集中在模块顶部，供 choose_route 与单测共享）
# ---------------------------------------------------------------------------
GEOMETRY_CORRELATION_MIN = 0.5
ZERO_SHIFT_PX_MAX = 20.0
MODEL_ANCHOR_ERROR_MAX = 0.25
SNR_MIN = 3.0


# ---------------------------------------------------------------------------
# 数据契约（与并行 lane 共享，逐字遵守）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeReport:
    """小规模探测报告：路由决策的输入。

    所有判据字段均可为 ``None``（未知）。``None`` 一律视为**未知而非坏**
    —— 不触发任何风险分支，只在不能被判好的证据缺失时才影响判定。
    """

    intent: str = "auto"        # auto|extended_field|new_mask|fixed_shape|uniformity|cold_start
    shape: str = "square"
    calibrated_shapes: tuple[str, ...] = ("square", "rectangle", "triangle")
    has_checkpoint: bool = False
    has_seed_run: bool = False
    light_ok: bool | None = None
    geometry_correlation: float | None = None   # 逐探针相关 r；<0.5 视为光路/几何失配
    zero_shift_px: float | None = None          # 零级相对上次标定的漂移
    direction_agreement: float | None = None    # 方向检验，负值=方向错
    focus_changed: bool | None = None
    model_anchor_error: float | None = None     # 冻结模型预测 vs 实测锚点相对偏差
    snr: float | None = None
    noise_floor_ok: bool | None = None
    geometry_problems: tuple[str, ...] = ()     # preflight.check_geometry 的 problem code
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoutePlan:
    """一条路由决策的输出：推荐的整形路线 + 验收红线 + 执行参数。"""

    route_id: str
    title: str                  # 中文路线名
    methods: tuple[str, ...]    # 如 ("#1", "#2", "#9")
    prerequisites: tuple[str, ...]
    forbidden: tuple[str, ...]
    reasons: tuple[str, ...]    # 由哪些判据推出，含具体数值
    red_lines: tuple[str, ...]  # 验收红线
    recommended: dict           # 建议执行方式（runner/脚本/参数覆盖），纯数据
    requires_confirmation: bool = True


# ---------------------------------------------------------------------------
# 各路线推荐执行参数（纯数据；与仓库现有入口对应）
# ---------------------------------------------------------------------------
_RECOMMENDED: dict[str, dict] = {
    "no_go": {
        "steps": ["降低噪声或提高信号", "重新探测"],
    },
    "extended_field": {
        "note": "仅超视场预测/物理项研究，独立确认门 + 指标选择偏差注明",
    },
    "refocus": {
        "steps": ["probe_session_focus", "measure_session_geometry",
                  "方向检验", "重标定后再按 stable_small"],
    },
    "cold_start": {
        "entry": "ao_shaping.tools.slm.transfer.workflow.run_session",
        "overrides": {"checkpoint": None, "initial_phase": "gs",
                      "accept_metric": "cv"},
    },
    "new_mask": {
        "steps": ["标定孪生闭环 finetune", "逐轮验收门拦坏候选",
                  "实测同曝光验收"],
    },
    "anchored_uniformity": {
        "entry": "ao_shaping.tools.slm.transfer.workflow.run_session",
        "overrides": {"objective": "cv", "accept_metric": "cv",
                      "adaptive_exposure": True, "abba": True},
    },
    "stable_small": {
        "runner": "scripts/ml/sim2f_local_forward_experiment.py",
        "protocol": "central-difference + 3轮有界更新 + A-B-B-A + 独立复测",
    },
}


def _route(
    route_id: str,
    title: str,
    methods: tuple[str, ...],
    prerequisites: tuple[str, ...],
    forbidden: tuple[str, ...],
    reasons: tuple[str, ...],
    red_lines: tuple[str, ...],
    recommended: dict,
    *,
    requires_confirmation: bool = True,
) -> RoutePlan:
    """构造一条 RoutePlan，推荐参数取自集中表并浅拷贝（防外部改写）。"""
    return RoutePlan(
        route_id=route_id,
        title=title,
        methods=methods,
        prerequisites=prerequisites,
        forbidden=forbidden,
        reasons=reasons,
        red_lines=red_lines,
        recommended=dict(recommended),
        requires_confirmation=requires_confirmation,
    )


def _router_collect_refocus_reasons(report):
    """Collect the refocus reasons for one report."""
    refocus_reasons: list[str] = []
    if report.light_ok is False:
        refocus_reasons.append("light_ok=False：光路未就绪")
    if report.geometry_correlation is not None:
        if report.geometry_correlation < GEOMETRY_CORRELATION_MIN:
            refocus_reasons.append(
                f"geometry_correlation={report.geometry_correlation:g}"
                f" < {GEOMETRY_CORRELATION_MIN}：光路/几何失配")
    if report.direction_agreement is not None and report.direction_agreement < 0:
        refocus_reasons.append(
            f"direction_agreement={report.direction_agreement:g} < 0：方向错")
    if report.focus_changed is True:
        refocus_reasons.append("focus_changed=True：焦面已变化")
    if report.zero_shift_px is not None and abs(report.zero_shift_px) > ZERO_SHIFT_PX_MAX:
        refocus_reasons.append(
            f"zero_shift_px={report.zero_shift_px:g}，|shift|"
            f" > {ZERO_SHIFT_PX_MAX}：零级漂移超限")
    if report.geometry_problems:
        refocus_reasons.append(
            f"geometry_problems={list(report.geometry_problems)}")
    return refocus_reasons


def _router_collect_anchor_reasons(report):
    """Collect the uniformity intent and anchor reasons."""
    uniformity_intent = (
        report.intent == "uniformity"
        or (report.intent == "fixed_shape"
            and report.shape in ("square", "rectangle"))
    )
    anchor_reasons: list[str] = []
    if report.model_anchor_error is not None and report.model_anchor_error > MODEL_ANCHOR_ERROR_MAX:
        anchor_reasons.append(
            f"model_anchor_error={report.model_anchor_error:g}"
            f" > {MODEL_ANCHOR_ERROR_MAX}：冻结模型锚点偏离")
    return anchor_reasons, uniformity_intent


def choose_route(report: ProbeReport) -> RoutePlan:
    """按固定优先级把探测报告映射为推荐路线。命中即定，不继续下探。"""
    reasons: list[str] = []

    # 1. no_go —— 优先级最高：噪声高于相位差效应，整形无意义。
    if report.noise_floor_ok is False:
        reasons.append("noise_floor_ok=False：噪声底高于相位差效应")
    if report.snr is not None and report.snr < SNR_MIN:
        reasons.append(f"snr={report.snr:g} < {SNR_MIN}：信噪比不足")
    if reasons:
        return _route(
            "no_go",
            "不整形（先整改噪声/信号）",
            (),
            ("降低噪声或提高信号，重跑探测", "确认信号真实后再探一次"),
            ("在噪声高于相位差效应时做任何整形",),
            tuple(reasons),
            ("禁止基于低于噪声底的指标下任何结论",
             "禁止把噪声当整形信号",
             "禁止沿用上轮模型与曝光"),
            dict(_RECOMMENDED["no_go"]),
        )

    # 2. extended_field —— 超视场建模/物理项研究，不作为工艺整形。
    if report.intent == "extended_field":
        return _route(
            "extended_field",
            "超视场/物理项建模",
            ("#13",),
            ("独立确认门：外推结果须在超视场点上单独验收",
             "指标选择偏差需在报告里注明"),
            ("用窗内（视场内）残差宣称超视场泛化",),
            ("intent=extended_field",),
            ("在外推视场点上有独立数据确认，不得用窗内残差替代",),
            dict(_RECOMMENDED["extended_field"]),
        )

    # 3. refocus —— 光路/焦面刚变化。任一证据成立即命中。
    refocus_reasons = _router_collect_refocus_reasons(report)
    if refocus_reasons:
        return _route(
            "refocus",
            "光路/焦面重对准（重标定后再整形）",
            ("#10", "#4", "#11"),
            ("焦面矫正（focus_probe）",
             "几何重标定（measure_session_geometry）",
             "方向检验通过后再回零级/窗口定位"),
            ("沿用上轮冻结模型", "沿用上轮曝光设置",
             "配合未重标定的几何任一参数直接整形"),
            tuple(refocus_reasons),
            ("重标定后零级/窗口回到参考位置",
             "几何相关 >=0.5 且方向检验非负",
             "重测 NoIl 前不透传上轮模型"),
            dict(_RECOMMENDED["refocus"]),
        )

    # 4. cold_start —— 无任何先验资料。
    if report.intent == "cold_start" or (not report.has_checkpoint
                                         and not report.has_seed_run):
        return _route(
            "cold_start",
            "冷启动（能量/匀化从零建）",
            ("#8", "#9"),
            ("能量优先目标（先拉开信噪比）",
             "CV/EE 成对评估",
             "同步曝光端点（同曝光比较）"),
            ("用训练损失挑选最佳相位（应以实测指标为准）",),
            ("intent=cold_start" if report.intent == "cold_start"
             else "无 checkpoint 且无 seed run"),
            ("CV/EE 在实测同曝光端点上成对校验",
             "训练损失不得作为验收依据"),
            dict(_RECOMMENDED["cold_start"]),
        )

    # 5. new_mask —— 新掩膜或未标定形状。
    if report.intent == "new_mask" or report.shape not in report.calibrated_shapes:
        return _route(
            "new_mask",
            "新掩膜/未标定形状",
            ("#5", "#9"),
            ("先在标定孪生上闭环验证",
             "闭环 finetune 后再考虑上新掩膜",
             "直接用未标定的出厂参数做配对",
             "逐轮坏候选拦截（验收门）"),
            ("网络开环相位直接下发到设备",),
            (f"shape={report.shape}"
             + ("" if report.shape in report.calibrated_shapes
                else f" 未在 calibrated_shapes={list(report.calibrated_shapes)}"),
             "intent=new_mask" if report.intent == "new_mask" else None),
            ("坏候选在逐轮验收门被拦下",
             "下发前经标定孪生验证过的相位才允许"),
            dict(_RECOMMENDED["new_mask"]),
        )

    # 6. anchored_uniformity —— 匀化意图，或冻结模型锚点偏离严重。
    anchor_reasons, uniformity_intent = _router_collect_anchor_reasons(report)
    if uniformity_intent or (anchor_reasons and report.shape
                             in ("square", "rectangle")):
        all_reasons = list(anchor_reasons)
        if uniformity_intent:
            all_reasons.append(
                "uniformity 意图" if report.intent == "uniformity"
                else f"fixed_shape + shape={report.shape}")
        return _route(
            "anchored_uniformity",
            "锚定匀化（以实测锚点为准）",
            ("#3", "#9"),
            ("以实测锚点冻结窗口/目标",
             "同曝光双档（自适应曝光）",
             "增量方向修正（不接受全零指标）"),
            ("把冻结模型预测 CV 当成实测验收值",),
            tuple(all_reasons),
            ("实测锚点与冻结模型锚点一致（或已按实测重定）",
             "双档同曝光的 CV 不再因曝光漂移虚高",
             "拒绝全零/近零级的全零指标"),
            dict(_RECOMMENDED["anchored_uniformity"]),
        )

    # 7. stable_small —— 默认路线。
    return _route(
        "stable_small",
        "稳定小幅整形",
        ("#1", "#2", "#9"),
        ("零级与目标窗口冻结",
         "以连续相位作为起点",
         "每轮重辨识（含几何重标定确认）",
         "独立复测（A-B-B-A）"),
        ("直接用未标定的出厂参数给步长",),
        ("默认落款：无风险证据命中",),
        ("拒绝零曝光/近零级的全零指标",
         "实测温漂前回零级重测"),
        dict(_RECOMMENDED["stable_small"]),
    )


def plan_to_dict(plan: RoutePlan) -> dict[str, Any]:
    """把 RoutePlan 转成可 JSON 序列化的 dict（元组/None 展开为纯格式）。"""
    return json.loads(json.dumps(asdict(plan)))


def format_plan(plan: RoutePlan) -> str:
    """中文人读摘要：路线/方法/依据/前置/禁止/红线/建议/需人工确认。"""
    lines = [
        f"路线：{plan.title}（{plan.route_id}）",
        f"方法：{'、'.join(plan.methods) if plan.methods else '（无，仅整改）'}",
    ]
    if plan.reasons:
        lines.append(f"依据：{'；'.join(plan.reasons)}")
    if plan.prerequisites:
        lines.append(f"前置：{'；'.join(plan.prerequisites)}")
    if plan.forbidden:
        lines.append(f"禁止：{'；'.join(plan.forbidden)}")
    if plan.red_lines:
        lines.append(f"红线：{'；'.join(plan.red_lines)}")
    lines.append(f"建议：{json.dumps(plan.recommended, ensure_ascii=False)}")
    lines.append("需人工确认：是" if plan.requires_confirmation else
                 "需人工确认：否")
    return "\n".join(lines)