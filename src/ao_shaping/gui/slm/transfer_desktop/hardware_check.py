"""迁移学习闭环桌面页的中立硬件自检层。

不 import Qt / Streamlit，只依赖标准库、相机驱动、``probe_slm`` 与
``configured_daheng_exposure_limit``，供 Qt 页在后台线程调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ao_shaping.drivers.ccd import DahengCamera, MIICamera
from ao_shaping.drivers.slm.santec.probe import probe_slm
from ao_shaping.utils.image.exposure_control import configured_daheng_exposure_limit

__all__ = ["CheckItem", "HardwareReport", "check_hardware"]


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


def _selected_cam_present(cam_list: object, cam_id: int) -> bool:
    """判断所选 cam_id 是否在枚举列表中。"""
    if cam_list is None:
        return False
    try:
        materialised = list(cam_list)
    except TypeError:  # policy: a non-iterable enumeration result is treated as "not present", not as a crash
        return False
    if not materialised:
        return False

    reported_indices: list[object] = []
    for item in materialised:
        idx = None
        if isinstance(item, dict):
            idx = item.get("index")
        else:
            idx = getattr(item, "index", None)
        if idx is not None:
            reported_indices.append(idx)

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
        try:
            cam_list = cam_cls.get_cam_list()
            cam_items = [] if cam_list is None else list(cam_list)
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
        except Exception as exc:  # policy: one backend failing must not abort the other checks
            items.append(
                CheckItem(
                    label=label,
                    level="fail" if required else "warn",
                    detail=(
                        f"{backend} 枚举失败：{type(exc).__name__}: {exc}"
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