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
    resolve_dm,
)
from ao_shaping.drivers.dm.hadamard_dm import HadamardDM
from ao_shaping.drivers.dm.zernike_dm import ZernikeDM
from ao_shaping.drivers.dm.micro import (
    MicroDM,
    MicroDMError,
    MicroDMConnectionError,
    MicroDMVoltageError,
)

# 仿真 DM 是惰性解析的。在此处导入它们会造成循环:
# 本 (硬件) 包 -> drivers.sim.dm -> simulated_micro_dm ->
# drivers.dm.base -> 又回到本模块, 而此时 drivers.sim.dm 只初始化了一半。
# 该循环此前被掩盖, 因为 ``drivers/__init__`` 会 eager 导入
# ``dm.nlight``, 在 ``drivers.sim`` 被触及之前就完成了本模块;
# 把那次导入改为惰性后就暴露了出来。
_LAZY_BACKENDS: dict[str, tuple[str, str]] = {
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
    "MicroDMVoltageError",
    "SimMicroDM",
    "SimulateDM",
    "ZernikeDM",
    "create_dm",
    "get_dm_registry",
    "list_dm_types",
    "list_reachable_dm_types",
    "register_dm",
    "resolve_dm",
]
