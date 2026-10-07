from typing import TYPE_CHECKING

from ao_shaping.drivers._lazy import install_lazy_attrs
from ao_shaping.drivers.dm.base import DM
from ao_shaping.drivers.dm._registry import (
    DMRegistry,
    create_dm,
    get_dm_registry,
    list_dm_types,
    list_reachable_dm_types,
    register_dm,
    register_dm_lazy,
    resolve_dm,
)

# 在模块作用域注册硬件驱动类型名, 不触发任何驱动模块的 import。
# 这使得 ``list_dm_types()`` 在零驱动加载的情况下就能返回完整列表。
#
# 单模块驱动 (hadamard_dm, zernike_dm) 没有自己的 ``__init__``,
# 所以注册调用放在这里。
register_dm_lazy("hadamard", "ao_shaping.drivers.dm.hadamard_dm", "HadamardDM")
register_dm_lazy("zernike", "ao_shaping.drivers.dm.zernike_dm", "ZernikeDM")

# 子包驱动 (micro, nlight) 的 ``register_dm_lazy`` 调用在各自 ``__init__`` 里,
# 这里只做 lightweight import 以触发那些注册副作用, 不导入任何驱动模块。
import ao_shaping.drivers.dm.micro  # noqa: E402,F401
import ao_shaping.drivers.dm.nlight  # noqa: E402,F401

# 仿真 DM 是惰性解析的。在此处导入它们会造成循环:
# 本 (硬件) 包 -> drivers.sim.dm -> simulated_micro_dm ->
# drivers.dm.base -> 又回到本模块, 而此时 drivers.sim.dm 只初始化了一半。
# 该循环此前被掩盖, 因为 ``drivers/__init__`` 会 eager 导入
# ``dm.nlight``, 在 ``drivers.sim`` 被触及之前就完成了本模块;
# 把那次导入改为惰性后就暴露了出来。
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
    # 硬件驱动类 — 每个子包自己在 ``__init__`` 里做了 ``register_dm_lazy``,
    # 这里只为 ``from ao_shaping.drivers.dm import XxxDM`` 这种旧路径
    # 保留属性级惰性解析。
    "HadamardDM": ("ao_shaping.drivers.dm.hadamard_dm", "HadamardDM"),
    "ZernikeDM": ("ao_shaping.drivers.dm.zernike_dm", "ZernikeDM"),
    "MicroDM": ("ao_shaping.drivers.dm.micro.driver", "MicroDM"),
    "MicroDMConnectionError": (
        "ao_shaping.drivers.dm.micro.driver",
        "MicroDMConnectionError",
    ),
    "MicroDMError": ("ao_shaping.drivers.dm.micro.driver", "MicroDMError"),
    "MicroDMParams": ("ao_shaping.drivers.dm.micro.driver", "MicroDMParams"),
    "MicroDMVoltageError": (
        "ao_shaping.drivers.dm.micro.driver",
        "MicroDMVoltageError",
    ),
    # 仿真 DM
    "SimMicroDM": ("ao_shaping.drivers.sim.dm", "SimMicroDM"),
    "SimulateDM": ("ao_shaping.drivers.sim.dm", "SimulateDM"),
}

if TYPE_CHECKING:
    from ao_shaping.drivers.sim.dm import SimMicroDM, SimulateDM


__getattr__ = install_lazy_attrs(globals(), _LAZY_BACKENDS, __name__)

__all__ = [
    "DM",
    "DMRegistry",
    "HadamardDM",
    "MicroDM",
    "MicroDMConnectionError",
    "MicroDMError",
    "MicroDMParams",
    "MicroDMVoltageError",
    "SimMicroDM",
    "SimulateDM",
    "ZernikeDM",
    "create_dm",
    "get_dm_registry",
    "list_dm_types",
    "list_reachable_dm_types",
    "register_dm",
    "register_dm_lazy",
    "resolve_dm",
]
