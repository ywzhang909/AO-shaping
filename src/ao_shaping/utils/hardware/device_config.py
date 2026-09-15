"""通用设备配置工具

提供配置参数的加载、检查、应用和保存功能，与具体的参数定义解耦。

核心组件:
    - DeviceParam: 参数 dataclass 基类
    - ConfigHandler[T]: 泛型配置处理器，处理加载→解析→应用→保存
    - param(): 字段标记函数，用于在 dataclass 字段上声明元数据

使用示例:
    >>> from dataclasses import dataclass
    >>> from ao_shaping.utils.device_config import DeviceParam, ConfigHandler, param
    >>>
    >>> @dataclass
    >>> class MyParams(DeviceParam):
    ...     wavelength: int | None = param(default=None, cast=int)
    ...     shift_x: int = param(default=0, attr="_shift_x")
    ...
    >>> handler = ConfigHandler("data/configs", "my_device", MyParams)
    >>> params = handler.apply(instance, serial, init_values={"wavelength": 1064})
    >>> handler.save(serial, instance)
"""

from __future__ import annotations

from dataclasses import MISSING, Field, dataclass, fields, field
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from loguru import logger

from ao_shaping.utils.file import DeviceConfigManager

T = TypeVar("T")


def param(
    default: Any = MISSING,
    *,
    config_key: str | None = None,
    cast: Callable[[Any], Any] | None = None,
    attr: str | None = None,
) -> Any:
    """标记 dataclass 字段为设备配置参数。

    用于 ``DeviceParam`` 子类的字段上，将字段与 JSON 配置 key
    以及设备实例属性关联。

    Args:
        default: 默认值（不传则字段无默认值）。
        config_key: JSON 配置文件中的键名。默认使用字段名。
        cast: 类型转换函数，例如 ``int``、``bool``、``MlaRes``。
        attr: 设备实例的属性名。默认使用字段名。
          当属性名与字段名不同时（如字段 ``shift_x`` → 属性 ``_shift_x``）使用。

    Returns:
        ``dataclasses.field()`` 返回值，带有元数据。
    """
    meta: dict[str, Any] = {}
    if config_key is not None:
        meta["config_key"] = config_key
    if cast is not None:
        meta["cast"] = cast
    if attr is not None:
        meta["attr"] = attr

    if default is not MISSING:
        return field(default=default, metadata=meta)
    return field(metadata=meta)


class DeviceParam:
    """设备参数基类

    所有设备参数的 dataclass 应继承此类。配合 ``param()`` 标记字段。
    """

    @classmethod
    def _field_meta(cls, f: Field) -> dict[str, Any]:
        """获取字段的配置元数据

        返回:
            包含 ``config_key``、``cast``、``attr`` 的字典。
        """
        m = f.metadata or {}
        return {
            "config_key": m.get("config_key", f.name),
            "cast": m.get("cast"),
            "attr": m.get("attr", f.name),
        }


class ConfigHandler(Generic[T]):
    """通用设备配置处理器

    将与具体设备无关的配置加载、检查、应用逻辑封装为泛型类。
    泛型参数 ``T`` 是继承 ``DeviceParam`` 的参数 dataclass。

    优先级规则（高→低）:
        1. ``__init__`` 显式传入的参数（init_values）
        2. JSON 配置文件（按序列号匹配）
        3. dataclass 字段默认值
    """

    def __init__(
        self,
        config_dir: str | Path,
        device_type: str,
        param_cls: type[T],
    ):
        """
        Args:
            config_dir: 配置文件根目录。
            device_type: 设备类型标识（如 ``"slm"``、``"wfs"``）。
            param_cls: 设备参数 dataclass 类型。
        """
        self._manager = DeviceConfigManager(config_dir, device_type)
        self._param_cls = param_cls

    def _resolve_value(
        self,
        config_key: str,
        attr: str,
        config: dict[str, Any],
        init_values: dict[str, Any],
        default: Any,
    ) -> Any:
        """按优先级解析某个参数的值。

        优先级: init_values > config > default。

        Args:
            config_key: JSON 配置 key。
            attr: 设备实例属性名。
            config: 配置文件原始字典。
            init_values: __init__ 时传入的显式参数字典。
            default: dataclass 字段的默认值。

        Returns:
            解析后的参数值。
        """
        if attr in init_values and init_values[attr] is not None:
            return init_values[attr]
        if config_key in config:
            return config[config_key]
        return default

    def resolve(
        self,
        serial: str,
        init_values: dict[str, Any] | None = None,
    ) -> T:
        """加载并解析配置文件，返回设备参数对象。

        Args:
            serial: 设备序列号。为空时跳过文件加载，仅用
                init_values + 默认值。
            init_values: __init__ 时传入的显式参数字典。

        Returns:
            填充完毕的 ``T`` 类型参数对象。
        """
        config = self._manager.load_config(serial) if serial else {}
        return self.resolve_from_config(config, init_values)

    def resolve_from_config(
        self,
        config: dict[str, Any],
        init_values: dict[str, Any] | None = None,
    ) -> T:
        """从已加载的配置字典解析参数对象（不重新加载文件）。

        Args:
            config: 已加载的配置字典（来自 ``self._manager.load_config()`` 等）。
            init_values: __init__ 时传入的显式参数字典。

        Returns:
            填充完毕的 ``T`` 类型参数对象。
        """
        init_values = init_values or {}

        kwargs: dict[str, Any] = {}
        for f in fields(self._param_cls):
            meta = self._param_cls._field_meta(f)
            key: str = meta["config_key"]
            attr: str = meta["attr"]
            cast_fn = meta["cast"]

            value = self._resolve_value(
                key, attr, config, init_values,
                f.default if f.default is not MISSING else None,
            )

            if cast_fn is not None and value is not None:
                try:
                    value = cast_fn(value)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        "参数 '{}' 类型转换 ({}) 失败: {}，使用默认值",
                        key, cast_fn.__name__, e,
                    )
                    value = f.default if f.default is not MISSING else None

            kwargs[f.name] = value

        return self._param_cls(**kwargs)

    def apply(
        self,
        instance: Any,
        serial: str,
        init_values: dict[str, Any] | None = None,
    ) -> T:
        """加载并解析配置文件，将结果应用到设备实例的属性上。

        等价于 ``resolve()`` + ``_set_params()``。

        Args:
            instance: 设备实例，其属性将被设置。
            serial: 设备序列号。
            init_values: __init__ 时传入的显式参数字典。

        Returns:
            填充完毕的参数对象（可读回所有已解析的值）。
        """
        params = self.resolve(serial, init_values)
        self._set_params(instance, params)
        return params

    def apply_from_config(
        self,
        instance: Any,
        config: dict[str, Any],
        init_values: dict[str, Any] | None = None,
    ) -> T:
        """使用已加载的配置字典将参数应用到设备实例（不重新加载文件）。

        Args:
            instance: 设备实例。
            config: 已加载的配置字典。
            init_values: __init__ 时传入的显式参数字典。

        Returns:
            填充完毕的参数对象。
        """
        params = self.resolve_from_config(config, init_values)
        self._set_params(instance, params)
        return params

    @staticmethod
    def _set_params(instance: Any, params: DeviceParam) -> None:
        """将参数对象的值设置到实例属性上。"""
        for f in fields(params.__class__):
            meta = params.__class__._field_meta(f)
            attr: str = meta["attr"]
            setattr(instance, attr, getattr(params, f.name))

    def collect(self, instance: Any) -> dict[str, Any]:
        """从设备实例收集当前参数值，构建配置字典。

        用于 ``save()`` 前的数据收集。

        Args:
            instance: 设备实例。

        Returns:
            配置字典（key 为 JSON 配置 key）。
        """
        config: dict[str, Any] = {}
        for f in fields(self._param_cls):
            meta = self._param_cls._field_meta(f)
            attr: str = meta["attr"]
            key: str = meta["config_key"]
            config[key] = getattr(instance, attr)
        return config

    def save(self, serial: str, instance: Any) -> None:
        """将当前参数值保存到 JSON 配置文件。

        Args:
            serial: 设备序列号。
            instance: 设备实例（从中收集参数值）。
        """
        config = self.collect(instance)
        self._manager.save_config(serial, config)
