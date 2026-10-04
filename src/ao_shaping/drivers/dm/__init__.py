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

# The simulated DMs are resolved lazily. Importing them here created a cycle:
# this (hardware) package -> drivers.sim.dm -> simulated_micro_dm ->
# drivers.dm.base -> back into this module, while drivers.sim.dm was only half
# initialised. The cycle used to be masked because ``drivers/__init__`` imported
# ``dm.nlight`` eagerly, completing this module before ``drivers.sim`` was
# touched; making that import lazy exposed it.
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
