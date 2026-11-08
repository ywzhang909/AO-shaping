"""迁移学习闭环桌面页的中立硬件自检层。

不 import Qt / Streamlit，只依赖标准库、相机驱动、``probe_slm`` 与
``configured_daheng_exposure_limit``，供 Qt 页在后台线程调用。

两种用途，都只读：:func:`check_hardware` 是显式自检 (枚举相机 + 探针 SLM)；
:func:`select_ccd` 是页面加载时的相机自动探测 —— **只枚举、不开相机、不碰 SLM**。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ao_shaping.drivers.ccd import DahengCamera, MIICamera
from ao_shaping.drivers.slm.santec.probe import probe_slm
from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

__all__ = [
    "CcdChoice",
    "CheckItem",
    "HardwareReport",
    "check_hardware",
    "select_ccd",
]

#: 自动探测只认这两个硬件后端 (``sim`` 等仿真后端没有「在线设备」可言)。
_CCD_BACKENDS: tuple[str, ...] = ("daheng", "miicam")
_CCD_LABELS: dict[str, str] = {"daheng": "Daheng", "miicam": "MiiCam"}


@dataclass(frozen=True)
class CheckItem:
    """单个自检项。"""

    label: str
    level: Literal["ok", "warn", "fail"]
    detail: str
    required: bool = True


@dataclass(frozen=True)
class HardwareReport:
    """硬件自检报告。"""

    items: tuple[CheckItem, ...]

    @property
    def ok(self) -> bool:
        """所有必需项都不是 fail。"""
        return all(not (item.required and item.level == "fail") for item in self.items)

    def summary(self) -> str:
        """一行中文结论。"""
        failed = [item for item in self.items if item.required and item.level == "fail"]
        if not failed:
            return "硬件自检通过"
        return f"硬件自检未通过（{len(failed)} 项必需项失败）"


def _materialised_cam_list(cam_list: object) -> list[object]:
    """把枚举结果落成列表; 不可迭代一律当作 0 台。"""
    if cam_list is None:
        return []
    try:
        return list(cam_list)
    except TypeError:  # policy: a non-iterable enumeration result is treated as "no cameras"
        return []


def _reported_cam_indices(items: Sequence[object]) -> list[object]:
    """枚举项自己声明的 ``index`` (dict 与属性两条路; 都没有就留空)。"""
    reported: list[object] = []
    for item in items:
        if isinstance(item, dict):
            idx = item.get("index")
        else:
            idx = getattr(item, "index", None)
        if idx is not None:
            reported.append(idx)
    return reported


def _selected_cam_present(cam_list: object, cam_id: int) -> bool:
    """判断所选 cam_id 是否在枚举列表中。"""
    materialised = _materialised_cam_list(cam_list)
    if not materialised:
        return False

    reported_indices = _reported_cam_indices(materialised)
    if reported_indices:
        if cam_id in reported_indices:
            return True
        # Some backends report 1-based metadata indices.
        if (cam_id + 1) in reported_indices:
            return True

    try:
        return 0 <= cam_id < len(materialised)
    except Exception:  # policy: defensive; membership must never raise
        return False


def _cam_id_for_list(items: Sequence[object], cam_id: int) -> int:
    """在一份枚举列表里把 cam_id 落成一台真实存在的机器。

    所选 id 还在列表里就保留 (操作员另有安排时不自作主张); 否则取第一台的
    ``index``; 连 index 都读不到 (老驱动只回裸列表) 就退回 0 —— 与
    :func:`_selected_cam_present` 的容错口径保持一致。
    """
    if not items:
        return int(cam_id)
    if _selected_cam_present(items, cam_id):
        return int(cam_id)
    for idx in _reported_cam_indices(items[:1]):
        try:
            return int(idx)  # type: ignore[arg-type]
        except (TypeError, ValueError):  # policy: 非数字 index 不可用, 退回列表首项
            break
    return 0


def _exc_summary(exc: BaseException, *, limit: int = 160) -> str:
    """一行异常摘要 (类型 + 消息): 压掉换行并限长, 只给界面上一行结论用。"""
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _counts_text(counts: dict[str, int]) -> str:
    """``Daheng 1 台 / MiiCam 0 台`` (顺序固定, 不随所选后端变)。"""
    return " / ".join(f"{_CCD_LABELS[kind]} {counts[kind]} 台" for kind in _CCD_BACKENDS)


def _enumerate_cam_list(cam_cls: object) -> tuple[list[object], str | None]:
    """枚举一个后端并落成列表, 返回 ``(相机列表, 失败摘要)``。

    失败 (驱动未加载 / SDK 初始化异常 / 枚举抛错) 一律只算该后端 0 台, 摘要
    交给调用方写进界面; 绝不向上抛 —— 两个调用点 (自检与加载时探测) 都靠它。
    """
    try:
        if cam_cls is None:
            raise RuntimeError("驱动未加载（SDK 不可用）")
        return _materialised_cam_list(cam_cls.get_cam_list()), None  # type: ignore[attr-defined]
    except Exception as exc:  # policy: one backend failing must not abort the other checks
        return [], _exc_summary(exc)


@dataclass(frozen=True)
class CcdChoice:
    """加载时自动探测得出的相机选型 (纯数据, 界面直接显示 ``detail``)。"""

    kind: str
    cam_id: int
    source: Literal["preferred", "switched", "unavailable"]
    daheng_count: int
    miicam_count: int
    detail: str


def select_ccd(preferred: str = "daheng", cam_id: int = 0) -> CcdChoice:
    """只读枚举两个相机后端, 挑出这次该用哪一个 (绝不抛, 也不打开相机)。

    规则: 优先后端有机器 → 保留 (``preferred``); 优先后端 0 台而另一个后端有机器
    → 改用那一台 (``switched``); 两边都 0 台 (或枚举就失败) → 原样保留
    (``unavailable``)。任何一次枚举异常都只算该后端 0 台, 摘要写进 ``detail``:
    加载时探测只是提示, 绝不能因此挡住后面的完整流程。
    """
    preferred = str(preferred)
    try:
        wanted = int(cam_id)
    except (TypeError, ValueError):  # policy: 坏 cam_id 退回 0, 结论不因一个数字失败
        wanted = 0

    lists: dict[str, list[object]] = {kind: [] for kind in _CCD_BACKENDS}
    counts: dict[str, int] = {kind: 0 for kind in _CCD_BACKENDS}
    failures: dict[str, str] = {}
    for kind in _CCD_BACKENDS:
        cam_cls = DahengCamera if kind == "daheng" else MIICamera
        lists[kind], failure = _enumerate_cam_list(cam_cls)
        if failure is not None:
            failures[kind] = failure
        counts[kind] = len(lists[kind])

    if preferred not in counts:
        # ``sim`` 等后端没有「在线设备」可言: 不猜, 原样保留。
        chosen_kind, chosen_id, source = preferred, wanted, "unavailable"
        detail = f"自动探测：{preferred} 无在线设备概念，保留 {preferred} cam_id={chosen_id}"
    elif counts[preferred]:
        chosen_kind = preferred
        chosen_id = _cam_id_for_list(lists[preferred], wanted)
        source = "preferred"
        detail = (
            f"自动探测：{_CCD_LABELS[preferred]} {counts[preferred]} 台，"
            f"已选 {preferred} cam_id={chosen_id}"
        )
    elif any(counts[kind] for kind in _CCD_BACKENDS if kind != preferred):
        chosen_kind = next(
            kind for kind in _CCD_BACKENDS if kind != preferred and counts[kind]
        )
        chosen_id = _cam_id_for_list(lists[chosen_kind], wanted)
        source = "switched"
        detail = f"自动探测：{_counts_text(counts)}，已改用 {chosen_kind} cam_id={chosen_id}"
    else:
        chosen_kind, chosen_id, source = preferred, wanted, "unavailable"
        detail = f"自动探测：未发现在线相机（{_counts_text(counts)}），保留 {preferred} cam_id={chosen_id}"

    for kind in _CCD_BACKENDS:
        if kind in failures:
            detail += f"（{_CCD_LABELS[kind]} 枚举失败：{failures[kind]}）"

    return CcdChoice(
        kind=chosen_kind,
        cam_id=chosen_id,
        source=source,  # type: ignore[arg-type]
        daheng_count=counts["daheng"],
        miicam_count=counts["miicam"],
        detail=detail,
    )


def check_hardware(camera: str, cam_id: int, slm_number: int) -> HardwareReport:
    """执行中立硬件自检，返回报告。相机与 SLM 检查相互隔离。"""
    items: list[CheckItem] = []

    # 1/2. 相机枚举: 所选后端为必需项, 另一后端仅供信息。
    #    miicam 已验证 (beam-quality 台面 + 迁移学习闭环接口), 选它时 MiiCam
    #    枚举成为必需项, Daheng 只作提示。
    for backend, cam_cls, label in (
        ("daheng", DahengCamera, "Daheng 枚举"),
        ("miicam", MIICamera, "MiiCam 枚举"),
    ):
        required = backend == str(camera)
        cam_items, failure = _enumerate_cam_list(cam_cls)
        if failure is None:
            count = len(cam_items)
            present = _selected_cam_present(cam_items, cam_id)
            if present:
                items.append(
                    CheckItem(
                        label=label,
                        level="ok",
                        detail=(
                            f"枚举到 {count} 台 {backend} 相机，所选 cam_id={cam_id} 在列表中"
                            + ("" if required else "（仅自检）")
                        ),
                        required=required,
                    )
                )
            else:
                items.append(
                    CheckItem(
                        label=label,
                        level="fail" if required else "warn",
                        detail=(
                            f"枚举到 {count} 台 {backend} 相机，所选 cam_id={cam_id} 不在列表中"
                            + ("" if required else "（仅自检；当前所选相机非此后端）")
                        ),
                        required=required,
                    )
                )
        else:
            items.append(
                CheckItem(
                    label=label,
                    level="fail" if required else "warn",
                    detail=(
                        f"{backend} 枚举失败：{failure}"
                        + ("" if required else "（仅自检；当前所选相机非此后端）")
                    ),
                    required=required,
                )
            )

    # 3. 所选 SLM（必需）
    probe_error: Exception | None = None
    device: dict | None = None
    try:
        device = probe_slm(slm_number)
    except Exception as exc:  # policy: probe is best-effort; a raising probe must not abort the report
        probe_error = exc
    if probe_error is not None:
        items.append(
            CheckItem(
                label="SLM 探测",
                level="fail",
                detail=f"SLM #{slm_number} 探测异常：{type(probe_error).__name__}（请重试或检查 SDK）",
                required=True,
            )
        )
    elif device is None:
        items.append(
            CheckItem(
                label="SLM 探测",
                level="fail",
                detail=f"未发现 SLM #{slm_number}（探针超时或设备未连接）",
                required=True,
            )
        )
    elif bool(device.get("in_use")):
        items.append(
            CheckItem(
                label="SLM 探测",
                level="fail",
                detail=f"SLM #{slm_number} 被占用（请关闭占用该设备的程序后重试）",
                required=True,
            )
        )
    else:
        serial = device.get("serial")
        items.append(
            CheckItem(
                label="SLM 探测",
                level="ok",
                detail=f"SLM #{slm_number} 可用，序列号 {serial if serial else '读取失败'}",
                required=True,
            )
        )

    # 4. 曝光安全上限（非必需）
    try:
        ceiling = configured_daheng_exposure_limit()
        if ceiling and ceiling > 0:
            items.append(
                CheckItem(
                    label="曝光安全上限",
                    level="ok",
                    detail=f"曝光安全上限已配置：{ceiling} ms",
                    required=False,
                )
            )
        else:
            items.append(
                CheckItem(
                    label="曝光安全上限",
                    level="warn",
                    detail="曝光安全上限未配置（运行时按画面自动探测；或设 AO_DAHENG_MAX_EXPOSURE_MS）",
                    required=False,
                )
            )
    except Exception as exc:  # policy: exposure ceiling read is best-effort and must not abort other checks
        items.append(
            CheckItem(
                label="曝光安全上限",
                level="warn",
                detail=f"曝光安全上限读取异常：{type(exc).__name__}（请检查环境变量）",
                required=False,
            )
        )

    return HardwareReport(items=tuple(items))