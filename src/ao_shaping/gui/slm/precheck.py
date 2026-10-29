"""campaign 开跑前的检查卡：六项**有出处**的结论，而不是六个绿勾。

为什么需要它
------------
``report/shape_trials/multishape_scale/multishape_scale_live_20261028.md`` §3.2 那次
16 段 / 112 块里，至少三个问题**在开跑前就能看出来**，当时都是跑到一半才发现：

1. 目标框与零级剔除圆盘重叠 5.61 %（200 px 档），CV 必须与该剔除面积一并读
   （规划报告 §2 勘误）；重叠面积靠 ``preflight.check_zero_overlap`` 算，不靠手感。
2. 落盘量与磁盘余量：整份 campaign 实测写出 14.7 GB，而同一台机器的余量是按会话变化的。
3. 复用一份旧模型 —— 它在**原来那束光**上拟合的，光路一变，第一个变暗的读数要等到
   整形段第一块才出现（``transfer_shapes_runner`` 的 ``light_check`` 就是为这条加的）。

所以每一项都是 :class:`Check`：``status`` 只是"要不要停"，``detail`` 是一句中文
结论，``evidence`` 是**判据与出处**（报告章节 / 常量名 / 阈值是谁提的）。没有
``evidence`` 的一项不许出现在卡片上 —— 否则操作员无法判断"这个绿勾凭什么"。

红黄绿的分工（与 ``campaign.CampaignSettings.blocking`` 同一分工）
------------------------------------------------------------------
- ``stop``：**结构性**错误（跑不了 / 会写满盘）。按钮直接禁用。
- ``warn``：读数口径或设备占用上的提醒，必须逐项勾选才能开跑
  （:meth:`CheckCard.acknowledged`）。
- ``unknown``：**没能判**（缺零级位置、缺模型、无历史）。它既不是通过也不是告警：
  静默降级成 ``pass`` 正是"没测就说没问题"的那种错，所以单独一档，不用勾选。
- ``pass``：判过且没问题。

纯函数边界
----------
本模块的判定层不 import streamlit、不 import torch、不开设备：计划页要在**没接设备**
的机器上把卡片画出来（这是 issue #215 的硬要求）。``render_check_card`` 是唯一的
渲染入口，streamlit 在函数体内导入，与 ``transfer_progress.render_progress`` 同一写法。

「光路有没有光」**不在这张卡上**，也是有意的：``light_guard.verify_light`` 要
``read_frame`` 与 ``set_exposure_ms``，是活相机的动作；把它塞进判卡会让"没连设备也能
先看一眼"这条前提失效。真正的判定发生在 runner 里（``capture.precheck_light`` /
整形段每块的曝光阶梯），失败时抛 ``OpticalPathInvalid``，由 ⑥ 把 ``report()`` 原文与
``cause`` 的处置方向直接印出来（离线判卡只能说"未知"，不能说"没有光"）。
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ao_shaping.tools.slm.transfer.campaign import (
    CampaignSettings,
    campaign_matrix,
    estimate_disk_gb,
    estimate_seconds,
)
from ao_shaping.tools.slm.transfer.preflight import (
    WINDOW_MARGIN_PX,
    check_zero_overlap,
)

__all__ = [
    "BUDGET_HOURS",
    "MODEL_QUALITY_PROPOSAL",
    "SATURATION_WARN_FRACTION",
    "STATUS_LABELS",
    "SUBJECT",
    "Check",
    "CheckCard",
    "build_check_card",
    "render_check_card",
]

CheckStatus = Literal["pass", "warn", "stop", "unknown"]

#: 卡片主语：出现在渲染标题与导出文件名里，改它要同时改页面标题。
SUBJECT: str = "多形状整形 campaign"

#: 设备占用预算（小时）。超了只 ``warn`` 不 ``stop``：这条检查问的是"今天还来不来得及"，
#: 不是"跑得对不对"。缺省 8 h —— 一份 112 块的 campaign 实测整形墙钟 2675.0 s
#: （``multishape_scale_wallclock_20261028.md:328``），8 h 是给"含 capture/fit/重放、
#: 中间还要停下来看读数"留的余量，不是从报告里推出来的物理量。
BUDGET_HOURS: float = 8.0

#: 重放梯度的饱和告警线：某块全幅峰值到满量程的 98 % 时，共同重放很可能削顶。
SATURATION_WARN_FRACTION: float = 0.98

#: issue #214 提案的模型验证质量下限（平均相关系数）。**不是**实测阈值：
#: 卡片只把它当 ``warn`` 线并在 ``evidence`` 里写明"阈值待实测"，绝不 ``stop``。
MODEL_QUALITY_PROPOSAL: float = 0.80

#: 状态词（中文显示用）。``unknown`` 用「未知」而不是「—」：这张卡片要能直接读成句子。
STATUS_LABELS: dict[str, str] = {
    "pass": "通过",
    "warn": "注意",
    "stop": "停止",
    "unknown": "未知",
}

#: 从坏到好的顺序。``unknown`` 排在 ``warn`` 之后：它不阻断，但比通过更需要人看一眼。
_SEVERITY: tuple[str, ...] = ("pass", "unknown", "warn", "stop")


@dataclass(frozen=True)
class Check:
    """一项检查。

    Attributes:
        key: 稳定标识（也是告警勾选的键）。
        label: 中文短名（表格第一列）。
        status: ``pass`` / ``warn`` / ``stop`` / ``unknown``。
        detail: 一句中文结论（可以直接读成句子）。
        evidence: 中文的判据与出处。**不可省**：没有出处的结论等于要求操作员信任一个数字。
    """

    key: str
    label: str
    status: CheckStatus
    detail: str
    evidence: str


@dataclass(frozen=True)
class CheckCard:
    """六项检查的合集，以及"能不能按开始"的判定。"""

    checks: tuple[Check, ...]

    @property
    def status(self) -> str:
        """最坏的一项的状态（``stop`` > ``warn`` > ``unknown`` > ``pass``）。"""
        return max((check.status for check in self.checks),
                   key=_SEVERITY.index, default="pass")

    @property
    def blocking(self) -> bool:
        """有任一 ``stop`` 项：按钮必须禁用。"""
        return any(check.status == "stop" for check in self.checks)

    @property
    def warnings(self) -> tuple[Check, ...]:
        """所有 ``warn`` 项 —— 每项需要一个勾选才能开跑。"""
        return tuple(check for check in self.checks if check.status == "warn")

    def acknowledged(self, flags: Mapping[str, bool]) -> bool:
        """所有 ``warn`` 项都已勾选。

        与 ``bench_settings`` 的手工覆盖同一契约：红停项直接禁用按钮，黄项必须由人
        逐项确认。``unknown`` 不要求勾选（它不是"已知的风险"），但也不当作通过 ——
        它只在本卡片上显示「未知」。
        """
        return all(bool(flags.get(check.key)) for check in self.warnings)

    def by_key(self, key: str) -> Check | None:
        """按 ``key`` 取一项（页面用它给勾选框配说明）。"""
        return next((check for check in self.checks if check.key == key), None)


# --------------------------------------------------------------------------- #
# 读盘助手（全部免设备、全部会降级）
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    """读一个 JSON，读不到/读坏了都返回 ``None``（预检卡片不许因为产物坏了而崩）。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _newest_campaign(campaigns: Sequence[Path]) -> Path | None:
    """``campaigns`` 里 ``journal.json`` 最新的那个目录（无则 ``None``）。"""
    stamped: list[tuple[float, Path]] = []
    for directory in campaigns:
        journal = Path(directory) / "journal.json"
        try:
            stamped.append((journal.stat().st_mtime, Path(directory)))
        except OSError:
            continue
    if not stamped:
        return None
    return max(stamped)[1]


def _target_center(campaigns: Sequence[Path], side: float) -> tuple[float, float] | None:
    """从历史 campaign 的 ``geometry_plan.json`` 取本尺寸的目标框中心。

    目标框不是"零级所在处"：它落在零级 + 载波位移上（``geometry_plan.json`` 的
    ``shift``），所以页面拿不到实测零点时**不能**假设两者同心 —— 同心假设在 200 px 档
    会把 5.61 % 算成 28 %。这里改用上一份 campaign 的规划几何：它是这台实验系统上
    唯一可用的目标框位置先验，取不到就返回 ``None``（卡片据此报 ``unknown``，
    而不是编一个中心）。
    """
    for directory in sorted((Path(c) for c in campaigns), key=lambda p: p.name, reverse=True):
        plan = _read_json(directory / "geometry_plan.json")
        if not isinstance(plan, dict):
            continue
        for cell in plan.get("cells") or ():
            if not isinstance(cell, dict):
                continue
            try:
                cell_side = float(cell.get("side"))
            except (TypeError, ValueError):
                continue
            if abs(cell_side - float(side)) > 1e-9:
                continue
            x_range, y_range = cell.get("target_x"), cell.get("target_y")
            if not (isinstance(x_range, (list, tuple)) and len(x_range) == 2):
                continue
            if not (isinstance(y_range, (list, tuple)) and len(y_range) == 2):
                continue
            return ((float(x_range[0]) + float(x_range[1])) / 2.0,
                    (float(y_range[0]) + float(y_range[1])) / 2.0)
    return None


def _manifest_peak_ratio(campaign: Path) -> tuple[float, int] | None:
    """本 campaign 里 ``peak / full_scale`` 的最大值与样本数（无数据返回 ``None``）。

    用 ``manifest.json`` 的行而不是 ``campaign_progress`` 的段视图：段视图只留
    ``peak_inside``/``peak_full`` 而把 ``full_scale`` 丢掉，而饱和问的正是
    「峰值离满量程还剩多少」。
    """
    best: float | None = None
    seen = 0
    for manifest in sorted(campaign.glob("shape/*/*/manifest.json")):
        rows = _read_json(manifest)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            peak, full = row.get("peak"), row.get("full_scale")
            try:
                peak_value, full_value = float(peak), float(full)
            except (TypeError, ValueError):
                continue
            if full_value <= 0:
                continue
            ratio = peak_value / full_value
            seen += 1
            best = ratio if best is None else max(best, ratio)
    return None if best is None else (best, seen)


def _journals_mentioning(campaigns: Sequence[Path], needle: str) -> list[Path]:
    """``journal.json`` 文本里出现 ``needle`` 的 campaign 目录。"""
    hits: list[Path] = []
    for directory in campaigns:
        journal = Path(directory) / "journal.json"
        try:
            text = journal.read_text(encoding="utf-8")
        except OSError:
            continue
        if needle in text:
            hits.append(Path(directory))
    return hits


# --------------------------------------------------------------------------- #
# 六项检查
# --------------------------------------------------------------------------- #
def _geometry_check(*, settings: CampaignSettings, target_xy: tuple[float, float] | None,
                    zero_xy: tuple[float, float] | None,
                    cam_size: int | None) -> Check:
    """几何自洽：零级剔除圆盘压进目标框多少 + 目标框放不放得进采样窗/相机窗口。"""
    side = float(settings.side)
    span = float(settings.settings["window_span"].value)
    zero_radius = float(settings.settings["zero_radius"].value)
    evidence = (
        f"判据：零级剔除圆盘（半径 {zero_radius:g} px，--zero-radius）∩ 目标方框"
        f"（边长 {side:g} px）的面积占比，preflight.check_zero_overlap 解析计算；"
        "阈值 5 % / 15 % 取自规划报告 §2 勘误（s1200 0.79 % / s600 3.14 % / s200 5.61 %）。"
        f"采样窗判据：目标半边长 + {WINDOW_MARGIN_PX:g} px 余量 ≤ --window-span {span:g} px。"
    )
    if zero_xy is None or target_xy is None:
        where = "零级位置未测" if zero_xy is None else "目标框位置未知"
        return Check(
            key="geometry", label="几何自洽", status="unknown",
            detail=(f"{where}：先跑光路探针再开跑"
                    "（scripts/slm/probe_optical_path_validity.py；或先跑一次本尺寸的 "
                    "plan 脚本产出 geometry_plan.json）"),
            evidence=evidence)

    overlap = check_zero_overlap(target_xy[0], target_xy[1], side, zero_xy,
                                 zero_radius=zero_radius)
    pct = overlap.pct
    if pct > 15.0:
        status: CheckStatus = "stop"
        head = f"零级剔除区吃掉目标框 {pct:.2f} %（>15 %）：换尺寸或挪目标框"
    elif pct > 5.0:
        status = "warn"
        head = (f"零级剔除区吃掉目标框 {pct:.2f} %（5–15 %）：CV 必须与该剔除面积"
                "一并读，否则会把「核心里有零级」读成「优化变差」")
    else:
        status = "pass"
        head = f"零级剔除区吃掉目标框 {pct:.2f} %（≤5 %）：{overlap.note}"

    # 目标框放不进采样窗: validate.recentre_window 会**裁掉**目标而不是报错。
    needed = side / 2.0 + WINDOW_MARGIN_PX
    if span < needed:
        status = "stop"
        head += (f"；并且采样窗半跨 {span:g} px < 需要的 {needed:g} px"
                 "（目标会被裁掉而不是报错）")
    elif cam_size is not None and int(cam_size) > 0 and side > float(cam_size):
        status = "stop"
        head += f"；并且目标边长 {side:g} px 超过相机窗口 {int(cam_size)} px"
    return Check(key="geometry", label="几何自洽", status=status, detail=head,
                 evidence=evidence)


def _disk_check(*, settings: CampaignSettings, out_dir: Path) -> Check:
    """磁盘预算：预计落盘 × 1.2 ≤ 可用余量。"""
    rows = campaign_matrix(settings.shapes)
    needed_gb = estimate_disk_gb(rows) * 1.2
    evidence = (
        f"判据：estimate_disk_gb（{len(settings.shapes)} 形状 × 4 阶段静态基线，"
        "单段 0.25/1.18/1.92/0.33 GB，来自 wallclock 报告 §6.3 的「修复后写出字节」表）"
        "× 1.2 余量，与 shutil.disk_usage 的可用字节比较。余量 20 % 是写入过程中的"
        "临时 npz / HDF5 缓冲，不是另行实测的数字。"
    )
    probe = Path(out_dir)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return Check(key="disk", label="磁盘预算", status="unknown",
                     detail=f"读不到 {probe} 的磁盘余量（{exc}）：落盘前请手工确认",
                     evidence=evidence)
    free_gb = usage.free / 1e9
    detail = (f"预计落盘 ≈{estimate_disk_gb(rows):.1f} GB（含 20 % 余量 "
              f"{needed_gb:.1f} GB），{probe} 所在盘可用 {free_gb:.1f} GB")
    if free_gb < needed_gb:
        return Check(key="disk", label="磁盘预算", status="stop",
                     detail=detail + "：余量不足，先清理或换 --out", evidence=evidence)
    if free_gb < needed_gb * 1.5:
        return Check(key="disk", label="磁盘预算", status="warn",
                     detail=detail + "：余量偏紧，跑到一半可能写满", evidence=evidence)
    return Check(key="disk", label="磁盘预算", status="pass", detail=detail,
                 evidence=evidence)


def _time_check(*, settings: CampaignSettings, budget_hours: float) -> Check:
    """时间预算：整形墙钟期望值 vs 操作员给的设备占用预算。"""
    rows = campaign_matrix(settings.shapes)
    low, expected, high = estimate_seconds(rows)
    budget_s = float(budget_hours) * 3600.0
    evidence = (
        "判据：estimate_seconds（总块数 × 24 s/块；112 块 → 2688 s，与实测整形墙钟 "
        "2675.0 s 差 0.5 %，multishape_scale_wallclock_20261028.md:328）× ±40 % 区间"
        f"（同报告逐块 10–35 s 的散布）。预算是设备占用提醒，不是正确性判据；"
        f"本页缺省 {BUDGET_HOURS:g} h。注意：capture / fit / 重放不计入该估算。"
    )
    detail = (f"整形墙钟 ≈{expected / 60:.0f} min（区间 {low / 60:.0f}–{high / 60:.0f} min），"
              f"预算 {float(budget_hours):g} h")
    if expected > budget_s:
        return Check(key="time", label="时间预算", status="warn",
                     detail=detail + "：超出预算，先确认设备能占这么久", evidence=evidence)
    return Check(key="time", label="时间预算", status="pass", detail=detail,
                 evidence=evidence)


def _model_check(*, model_dir: Path | None) -> Check:
    """模型新鲜度：报告 ``fit/result.json`` 里存下的验证质量，不自造阈值。"""
    evidence = (
        "判据：<模型目录>/fit/result.json 的 metrics[]（每次验证的 correlation / "
        "relative_l2），取相关系数均值原样报出。0.80 是 issue #214 的**提案**，"
        "尚未在实验系统上实测标定，所以低于它只报警不阻断（阈值待实测）；"
        "没有 fit/result.json 说明该目录不是一份可用模型（或拟合没跑完）。"
    )
    if model_dir is None:
        return Check(key="model", label="模型新鲜度", status="unknown",
                     detail="未提供模型目录：将从零重训模型，多约 20 min 设备时间",
                     evidence=evidence)
    result_path = Path(model_dir) / "fit" / "result.json"
    result = _read_json(result_path)
    if not isinstance(result, dict):
        return Check(key="model", label="模型新鲜度", status="unknown",
                     detail=f"{result_path} 不存在或不可读：这份目录不能当模型复用",
                     evidence=evidence)
    metrics = [row for row in (result.get("metrics") or ()) if isinstance(row, dict)]
    grains = [float(row["correlation"]) for row in metrics
              if isinstance(row.get("correlation"), (int, float))]
    if not grains:
        return Check(key="model", label="模型新鲜度", status="unknown",
                     detail=f"{result_path} 里没有 correlation 指标：无法判断质量",
                     evidence=evidence)
    mean_corr = sum(grains) / len(grains)
    written = str(result.get("timestamp_utc") or "时间未记录")
    detail = (f"{Path(model_dir).name}：验证相关系数均值 {mean_corr:.3f}"
              f"（{len(grains)} 项，拟合迭代 {result.get('iteration', '—')}，写于 {written}）")
    if mean_corr < MODEL_QUALITY_PROPOSAL:
        return Check(key="model", label="模型新鲜度", status="warn",
                     detail=detail + f"：低于提案线 {MODEL_QUALITY_PROPOSAL:g}（阈值待实测）",
                     evidence=evidence)
    return Check(key="model", label="模型新鲜度", status="pass", detail=detail,
                 evidence=evidence)


def _saturation_check(*, recent_campaigns: Sequence[Path]) -> Check:
    """饱和预估：上一份 campaign 的全幅峰值还剩多少满量程。"""
    evidence = (
        "判据：最近一份 campaign 的 shape/<shape>/<stage>/manifest.json 里 "
        "peak / full_scale 的最大值。共同重放要求在**同一曝光**下重放所有位深，"
        "所以峰值最接近满量程的那一块决定梯度会不会削顶（--replay-exposures-auto "
        "就是按这条推导的，transfer/protocol.derive_replay_exposures）。"
        f"告警线 {SATURATION_WARN_FRACTION:.0%} 满量程。"
    )
    newest = _newest_campaign(recent_campaigns)
    if newest is None:
        return Check(key="saturation", label="饱和预估", status="unknown",
                     detail="无历史峰值，首次运行按每档重放实测", evidence=evidence)
    band = _manifest_peak_ratio(newest)
    if band is None:
        return Check(key="saturation", label="饱和预估", status="unknown",
                     detail=f"{newest.name} 的 manifest 里没有 peak / full_scale：无法预估",
                     evidence=evidence)
    ratio, seen = band
    detail = (f"{newest.name}：{seen} 行里最高全幅峰值到满量程的 {ratio:.1%}"
              "（据此推导共同重放曝光）")
    if ratio >= SATURATION_WARN_FRACTION:
        return Check(key="saturation", label="饱和预估", status="warn",
                     detail=detail + "：重放很可能削顶，先降曝光或改自动梯度",
                     evidence=evidence)
    return Check(key="saturation", label="饱和预估", status="pass", detail=detail,
                 evidence=evidence)


def _exposure_check(*, recent_campaigns: Sequence[Path]) -> Check:
    """曝光收敛史：历史 journal 里出现过 ``exposure did not converge`` 就提醒重测阶梯。"""
    needle = "exposure did not converge"
    evidence = (
        "判据：历史 campaign 的 journal.json 文本里是否出现 "
        "validate.run 抛出的 RuntimeError(\"exposure did not converge\")"
        "（validate.py 的曝光阶梯在 12 步内打不开窗口）。出现一次就说明这条光路的"
        "亮度窗口变了，重放/整形会先空转 ~48 s 再失败（light_guard 模块头）。"
    )
    if not recent_campaigns:
        return Check(key="exposure", label="曝光收敛史", status="unknown",
                     detail="没有历史 campaign：无曝光收敛记录可查", evidence=evidence)
    hits = _journals_mentioning(recent_campaigns, needle)
    if hits:
        names = "、".join(sorted(path.name for path in hits))
        return Check(key="exposure", label="曝光收敛史", status="warn",
                     detail=f"{names} 里记录过「曝光阶梯不收敛」：先重测曝光阶梯再开跑",
                     evidence=evidence)
    return Check(key="exposure", label="曝光收敛史", status="pass",
                 detail=f"已查 {len(recent_campaigns)} 份历史 journal：没有不收敛记录",
                 evidence=evidence)


def build_check_card(*, settings: CampaignSettings, out_dir: Path,
                     model_dir: Path | None,
                     zero_xy: tuple[float, float] | None,
                     recent_campaigns: Sequence[Path] = (),
                     target_xy: tuple[float, float] | None = None,
                     cam_size: int | None = None,
                     budget_hours: float = BUDGET_HOURS) -> CheckCard:
    """把六项检查算成一张卡片。

    Args:
        settings: 同一次 campaign 的派生参数集（形状 + 尺寸为输入）。
        out_dir: ``--out`` 根目录（磁盘预算在它所在的盘上算）。
        model_dir: 复用的模型目录；``None`` = 从零重训。
        zero_xy: 实测零级位置（sensor px）；``None`` = 未测（几何检查将报 ``unknown``）。
        recent_campaigns: 历史 campaign 目录（饱和预估取最新一份，曝光史扫全部）。
        target_xy: 目标框中心（sensor px）。不给时取历史 ``geometry_plan.json`` 里
            同尺寸的规划中心；再取不到就报 ``unknown``（同心假设在 200 px 档会把
            5.61 % 算成 28 %，所以宁可说"不知道"）。
        cam_size: 相机读出窗口边长（px，0/``None`` = 未开裁窗 → 不做该检查）。
        budget_hours: 设备占用预算（h），超过只 ``warn``。

    Returns:
        六项检查的 :class:`CheckCard`（顺序固定：几何、磁盘、时间、模型、饱和、曝光）。
    """
    center = target_xy if target_xy is not None else _target_center(recent_campaigns,
                                                                   settings.side)
    return CheckCard(checks=(
        _geometry_check(settings=settings, target_xy=center, zero_xy=zero_xy,
                        cam_size=cam_size),
        _disk_check(settings=settings, out_dir=out_dir),
        _time_check(settings=settings, budget_hours=budget_hours),
        _model_check(model_dir=model_dir),
        _saturation_check(recent_campaigns=recent_campaigns),
        _exposure_check(recent_campaigns=recent_campaigns),
    ))


# --------------------------------------------------------------------------- #
# 渲染（唯一的 streamlit 入口，函数体内导入）
# --------------------------------------------------------------------------- #
def render_check_card(card: CheckCard, *, key_prefix: str,
                      flags: Mapping[str, bool] | None = None) -> dict[str, bool]:
    """把卡片画成表 + 每个 ``warn`` 项一个勾选框，返回最新的勾选状态。

    ``warn`` 项逐个勾选才允许开跑（:meth:`CheckCard.acknowledged`），红停项不给勾选框：
    它不是"确认后仍要跑"的东西，是按钮直接禁用（与 ``bench_settings`` 的手工覆盖同一
    契约）。勾选框的 key 用 ``{key_prefix}_ack_{check.key}``，所以历史页与运行页可以
    各自持有独立的确认状态。

    Args:
        card: :func:`build_check_card` 的结果。
        key_prefix: 本页/本区块的 widget key 前缀。
        flags: 上一轮的勾选状态（缺省 = 全未勾选）。

    Returns:
        ``{check.key: bool}``：本轮勾选状态。
    """
    import streamlit as st

    previous = dict(flags or {})
    st.dataframe(
        [{"检查": check.label,
          "状态": STATUS_LABELS.get(check.status, check.status),
          "结论": check.detail,
          "判据/出处": check.evidence} for check in card.checks],
        hide_index=True, use_container_width=True,
    )
    for check in card.checks:
        if check.status == "stop":
            st.error(f"{check.label}：{check.detail}")
    updated: dict[str, bool] = {}
    for check in card.warnings:
        updated[check.key] = st.checkbox(
            f"已确认 · {check.label}：{check.detail}",
            value=bool(previous.get(check.key)),
            key=f"{key_prefix}_ack_{check.key}",
        )
    return updated
