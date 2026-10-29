"""相机像元尺寸 (pixel size) 的单一来源。

像元尺寸是**型号属性**, 不是序列号属性: 同一型号的多台相机共用同一个值。
因此它不写进按序列号存储的 ``CCDParams`` 配置, 而是登记在按型号索引的
``<CCD_CONFIG_DIR>/ccd/models.json``::

    {
      "MER2-507-23GM": {"pixel_size_um": 2.2}
    }

解析顺序 (``resolve_model_pixel_size_um``):

1. 型号配置文件 ``models.json``;
2. 本模块的已知型号表 ``KNOWN_MODEL_PIXEL_SIZE_UM`` (历史口径, 只作回落);
3. 两者都没有 ⇒ 抛 :class:`CameraError`, 错误信息里给出登记命令。

**绝不猜一个数**: 像元尺寸直接决定远场物理标度 (像素 ↔ 空间频率), 猜错 10% 会让
焦距、束腰、目标边长的所有换算一致地偏 10%, 而没有任何一处报错。

本模块是零导入叶子: 不 import 任何相机 SDK, 模块作用域不做磁盘 I/O
(``import ao_shaping.drivers.ccd`` 不得读盘)。写盘只发生在
``store_model_pixel_size_um`` 被显式调用时。

命令行 (缺登记时按提示录入)::

    python -m ao_shaping.drivers.ccd.pixel_size              # 列出已登记型号
    python -m ao_shaping.drivers.ccd.pixel_size --prompt     # 询问当前连接的相机型号
    python -m ao_shaping.drivers.ccd.pixel_size --store MER2-507-23GM 2.2
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from ao_shaping.drivers.ccd.base import CameraError
from ao_shaping.utils.io.file import ROOT_DIR as PROJECT_ROOT

__all__ = [
    "KNOWN_MODEL_PIXEL_SIZE_UM",
    "camera_pixel_size_um_for",
    "ensure_model_pixel_size_um",
    "load_model_pixel_size_um",
    "lookup_model_pixel_size_um",
    "model_pixel_size_config_path",
    "probe_daheng_models",
    "resolve_model_pixel_size_um",
    "store_model_pixel_size_um",
]

#: 历史口径的已知型号表 (型号前缀 → 像元边长 µm)。
#: 只作 ``models.json`` 缺失时的回落, 不是权威来源 —— 权威是型号配置文件,
#: 因为同一型号前缀下可能有不同版本, 而配置里记的是实测确认过的值。
KNOWN_MODEL_PIXEL_SIZE_UM: dict[str, float] = {"MER2-507": 2.2}

#: 型号配置在 ``models.json`` 里的键名。
_CONFIG_KEY = "pixel_size_um"


def _config_dir() -> Path:
    """相机配置根目录 (与 ``DahengCamera`` 的 ``CCD_CONFIG_DIR`` 同一口径)。"""
    return Path(os.environ.get("CCD_CONFIG_DIR", PROJECT_ROOT / "data" / "ccd_configs"))


def model_pixel_size_config_path() -> Path:
    """型号配置文件路径: ``<CCD_CONFIG_DIR>/ccd/models.json``。"""
    return _config_dir() / "ccd" / "models.json"


def _read_models() -> dict[str, dict[str, float]]:
    """读取型号配置字典; 文件缺失或损坏时返回空字典 (损坏会记 warning)。"""
    path = model_pixel_size_config_path()
    if not path.exists():
        return {}
    try:
        # ``utf-8-sig``: 手写的配置常带 BOM (Windows 记事本/PowerShell 都默认加)。
        # 用 ``utf-8`` 读会让 ``json.load`` 报 "Unexpected UTF-8 BOM" 并把整份配置
        # 当未登记 —— 权威来源就此静默失效, 回落成已知型号表里的旧值。
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"相机型号配置读取失败 ({path}): {exc}; 按未登记处理")
        return {}
    if not isinstance(raw, dict):
        logger.warning(f"相机型号配置格式异常 ({path}): 顶层应为 dict; 按未登记处理")
        return {}
    out: dict[str, dict[str, float]] = {}
    for model, entry in raw.items():
        if isinstance(entry, dict):
            out[str(model)] = entry
    return out


def load_model_pixel_size_um(model: str) -> float | None:
    """从型号配置文件读取像元边长 (µm); 未登记返回 ``None``。"""
    entry = _read_models().get(str(model))
    if not entry:
        return None
    value = entry.get(_CONFIG_KEY)
    try:
        pixel = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning(
            f"相机型号 {model!r} 的 {_CONFIG_KEY} 不是数值 ({value!r}); 按未登记处理"
        )
        return None
    return pixel if pixel > 0.0 else None


def _lookup_known(model: str) -> float | None:
    """按已知型号表回落 (先精确匹配, 再取最长的型号前缀匹配)。"""
    name = str(model)
    if name in KNOWN_MODEL_PIXEL_SIZE_UM:
        return KNOWN_MODEL_PIXEL_SIZE_UM[name]
    hits = [k for k in KNOWN_MODEL_PIXEL_SIZE_UM if name.startswith(k)]
    if not hits:
        return None
    return KNOWN_MODEL_PIXEL_SIZE_UM[max(hits, key=len)]


def store_model_pixel_size_um(model: str, pixel_size_um: float) -> Path:
    """把型号的像元边长写入 ``models.json`` (合并写, 不覆盖其它型号)。

    Args:
        model: 相机型号 (SDK 的 ``DeviceModelName``, 如 ``MER2-507-23GM``)。
        pixel_size_um: 像元边长, 单位 µm, 必须为正。

    Returns:
        写入的配置文件路径。

    Raises:
        ValueError: 型号为空或像元边长非正。
    """
    name = str(model).strip()
    if not name:
        raise ValueError("相机型号为空, 无法登记像元尺寸")
    pixel = float(pixel_size_um)
    if not (pixel > 0.0):
        raise ValueError(f"像元尺寸必须为正 (µm), 收到 {pixel_size_um!r}")

    path = model_pixel_size_config_path()
    models = _read_models()
    models.setdefault(name, {})[_CONFIG_KEY] = pixel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
    logger.info(f"相机型号 {name} 的像元尺寸已登记: {pixel:g} µm → {path}")
    return path


def lookup_model_pixel_size_um(model: str) -> float | None:
    """型号配置 → 已知型号表; 都没有返回 ``None`` (不抛错)。

    供「探不到就当没有」的调用方使用 (如 GUI 侧栏的探测); 需要硬失败的路径请用
    :func:`resolve_model_pixel_size_um`。
    """
    name = str(model)
    pixel = load_model_pixel_size_um(name)
    if pixel is not None:
        return pixel
    return _lookup_known(name)


def resolve_model_pixel_size_um(model: str) -> float:
    """解析型号的像元边长 (µm): 配置文件 → 已知型号表 → 抛错。

    Raises:
        CameraError: 两处都没有登记。错误信息给出可直接执行的登记命令。
    """
    name = str(model)
    pixel = lookup_model_pixel_size_um(name)
    if pixel is not None:
        return pixel
    path = model_pixel_size_config_path()
    raise CameraError(
        f"相机型号 {name!r} 的像元尺寸未登记 (配置文件 {path} 与已知型号表都没有)。"
        f"请先登记: python -m ao_shaping.drivers.ccd.pixel_size "
        f"--store {name!r} <像元边长µm>, 或运行 --prompt 按提示录入。"
    )


def ensure_model_pixel_size_um(
    model: str,
    *,
    prompt: Callable[[str], str] | None = None,
) -> float:
    """已登记则直接返回; 否则询问用户、写入配置并返回。

    这是**脚本**侧的入口 (驱动只报错, 不交互): 需要像元尺寸的脚本在
    :class:`CameraError` 之前调用本函数, 用户输入一次之后, 同型号的后续运行
    直接命中配置。

    Args:
        model: 相机型号。
        prompt: 提问函数, 默认 ``input``; 传入不可交互的替代实现便于离线测试。

    Raises:
        CameraError: 用户未给出可用数值 (空输入 / 非数值 / 非正)。
    """
    try:
        return resolve_model_pixel_size_um(model)
    except CameraError:
        pass
    ask = prompt if prompt is not None else input
    answer = ask(
        f"相机型号 {model!r} 的像元尺寸 (µm) 未登记, 请输入 (回车放弃): "
    ).strip()
    try:
        pixel = float(answer)
    except ValueError:
        pixel = 0.0
    if not (pixel > 0.0):
        raise CameraError(
            f"相机型号 {model!r} 的像元尺寸未登记, 且未提供有效数值 ({answer!r})"
        )
    store_model_pixel_size_um(model, pixel)
    return pixel


def probe_daheng_models() -> list[str]:
    """列出当前可连接的大恒相机型号 (无 SDK 或无设备时返回空列表)。

    只枚举设备, **不打开**相机 —— 型号已经足够定位像元尺寸 (型号级配置),
    打开相机只为读一个静态属性既慢又会抢设备锁。
    """
    try:
        import gxipy as gx  # type: ignore[import-untyped]
    except (ImportError, OSError):
        return []
    try:
        _, dev_info_list = gx.DeviceManager().update_device_list()
    # 枚举失败 (GigE 驱动/网卡异常、设备被别的进程独占) 与"没有相机"对调用方的
    # 后果相同 —— 拿不到型号。gxipy 的异常类型只能在该 SDK 被导入后才知道 (本模块
    # 是零导入叶子), 收窄成 OSError 会把可用回落变成崩溃。
    # policy: 宽捕获归一成"探不到相机", 但必须留日志, 否则这句归一就是静默的。
    except Exception as exc:
        logger.warning(
            f"大恒相机枚举失败 ({type(exc).__name__}: {exc}); 按没有可连接相机处理"
        )
        return []
    models: list[str] = []
    for info in dev_info_list or []:
        raw = info.get("model", "") or ""
        if isinstance(raw, tuple):
            raw = raw[0] if raw else ""
        name = str(raw).strip()
        if name and name not in models:
            models.append(name)
    return models


def _probe_daheng_model_at(cam_id: int) -> str | None:
    """第 ``cam_id`` 台可连接大恒相机的型号名 (无设备返回 ``None``)。"""
    models = probe_daheng_models()
    if cam_id < 0 or cam_id >= len(models):
        return None
    return models[cam_id]


def camera_pixel_size_um_for(
    cam_type: str = "daheng",
    cam_id: int = 0,
    *,
    camera: Any | None = None,
    prompt: Callable[[str], str] | None = None,
    strict: bool = True,
) -> float:
    """按相机类型 (或已打开的相机实例) 解析像元尺寸 (µm)。

    这是脚本 / runner 的统一入口: 命令行把「未指定」表示为 ``0``, 由本函数变成
    真实值。解析顺序:

    1. ``camera`` 给出且已打开 ⇒ 直接读 ``camera.pixel_size_um`` (驱动是单一来源);
    2. ``daheng`` ⇒ 枚举设备取型号 (**不打开相机**) → 型号配置 → 已知型号表;
    3. ``miicam`` ⇒ ``MIICamera.pixel_size_um`` (写死 2.9);
    4. ``sim`` ⇒ ``SimPibCCD.pixel_size_um`` (孪生口径);
    5. 其它后端 ⇒ 抛错 (视频/图集回放没有可靠的物理标度), ``strict=False`` 时返回 0。

    Args:
        cam_type: 相机类型名 (``daheng`` / ``miicam`` / ``sim`` ...)。
        cam_id: 设备索引 (与大恒设备列表顺序一致)。
        camera: 已打开的相机实例; 给出时优先读它, 忽略 ``cam_type`` / ``cam_id``。
        prompt: 型号未登记时的提问函数。``None`` ⇒ 仅在交互终端 (stdin 是 TTY)
            才提问; 非交互环境直接抛驱动侧的 :class:`CameraError`。
        strict: 解析不到时抛错 (默认); ``False`` 则返回 ``0``。

    Raises:
        CameraError: ``strict=True`` 且该型号/后端没有可用的像元尺寸。
    """
    try:
        if camera is not None:
            return camera.require_pixel_size_um()
        name = str(cam_type or "").strip().lower()
        if name == "daheng":
            model = _probe_daheng_model_at(int(cam_id))
            if model is None:
                raise CameraError(
                    f"未找到第 {cam_id} 台大恒相机 (或 SDK 不可用), 无法确定像元尺寸; "
                    "请显式给出 --camera-pixel-um"
                )
            try:
                return resolve_model_pixel_size_um(model)
            except CameraError:
                ask = (
                    prompt
                    if prompt is not None
                    else (input if sys.stdin.isatty() else None)
                )
                if ask is None:
                    raise
                return ensure_model_pixel_size_um(model, prompt=ask)
        if name == "miicam":
            from ao_shaping.drivers.ccd.miicam.driver import MIICamera

            return float(MIICamera.pixel_size_um)
        if name == "sim":
            from ao_shaping.drivers.sim.slm_pib_sim import SimPibCCD

            return float(SimPibCCD.pixel_size_um)
        raise CameraError(f"相机类型 {cam_type!r} 没有可用的像元尺寸")
    except (CameraError, ImportError, OSError):
        if strict:
            raise
        return 0.0


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="相机像元尺寸登记 (型号级, 写入 <CCD_CONFIG_DIR>/ccd/models.json)"
    )
    parser.add_argument("--store", nargs=2, metavar=("MODEL", "UM"), help="登记型号像元尺寸")
    parser.add_argument(
        "--prompt", action="store_true", help="询问当前连接的大恒相机型号并登记"
    )
    args = parser.parse_args(argv)

    if args.store:
        model, value = args.store
        store_model_pixel_size_um(model, float(value))

    if args.prompt:
        models = probe_daheng_models()
        if not models:
            print("未发现可连接的大恒相机; 请用 --store MODEL UM 手工登记")
            return 1
        for model in models:
            ensure_model_pixel_size_um(model)

    print(f"配置文件: {model_pixel_size_config_path()}")
    registered = _read_models()
    for model in sorted(set(registered) | set(KNOWN_MODEL_PIXEL_SIZE_UM)):
        source = "配置文件" if model in registered else "已知型号表"
        value = load_model_pixel_size_um(model) or _lookup_known(model)
        print(f"  {model:<24} {value!s:>6} µm  ({source})")
    return 0


if __name__ == "__main__":  # pragma: no cover - 手工诊断入口
    raise SystemExit(_main())
