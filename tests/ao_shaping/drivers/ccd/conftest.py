r"""共享 SDK 引导: 让本目录下的硬件门控 CCD 测试不再因 ``gxipy`` 不在
``PYTHONPATH`` 而假性 skip / ``gx is None``。

本目录的 ``test_ccd.py`` / ``test_exposure_states.py`` 用 ``AO_RUN_HARDWARE=1``
门控, 并在模块作用域 ``pytest.importorskip("gxipy")`` 或经 ``create_camera("daheng")``
拉起后端。大恒驱动在**模块导入时**吞掉 ``import gxipy`` 的失败 (留下 ``gx = None``),
于是当随包 ``libs/gxipy`` 未在 ``PYTHONPATH`` 上时, 会**假性 skip** 或在构造期抛
``CameraError: gxipy SDK 不可用`` —— 相机明明在线、SDK 就在仓库 ``libs/`` 里。

``test_full_scale_hardware.py`` 自己先注入路径再 reload 所以不受影响。本 conftest 必须
**在导入 ``ao_shaping.drivers.ccd.daheng`` 包之前**就把随包 SDK 目录注入 ``sys.path``
—— 因为 ``daheng/__init__.py`` 会 eager import ``driver.py``, 而 ``driver.py`` 的模块级
``import gxipy`` 一旦在路径注入前执行, ``gx`` 就会锁死为 ``None`` (随后 reload 才能救回)。
故这里内联最小路径注入 (与 :func:`discovery._inject_sdk_path` 同口径), 再 import discovery
并 reload 一次 driver 兜底。

🔴 纯路径/导入操作, 不打开任何设备: ``discovery`` 是零导入叶子, ``bootstrap_sdk``
只 ``sys.path.insert`` + ``import gxipy``, 不触碰相机。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _inject_bundled_sdk_path() -> bool:
    """把随包 ``libs/gxipy`` 的父目录注入 ``sys.path``; 找不到合法绑定返回 ``False``。

    与 ``ao_shaping.drivers.ccd.daheng.discovery._inject_sdk_path`` 同口径, 但在此内联
    是为了让注入**早于** ``daheng`` 包的 eager import (见模块 docstring)。
    """
    raw = os.environ.get("AO_SHAPING_SDK_PATH")
    libs = Path(raw) if raw else Path(__file__).resolve().parents[4] / "libs"
    if not (libs / "gxipy" / "__init__.py").is_file():
        return False
    if str(libs) not in sys.path:
        sys.path.insert(0, str(libs))
    return True


# 先注入路径, 再 import 任何 ao_shaping.drivers.ccd.daheng.* —— 顺序不可颠倒。
_inject_bundled_sdk_path()

from ao_shaping.drivers.ccd.daheng import discovery  # noqa: E402

# 兜底: 若 driver 仍因更早的 import 顺序停在 gx=None, reload 一次 (与
# test_full_scale_hardware._load_driver 同款)。不可用时保持原状, 由各测试门控决定 skip。
discovery.bootstrap_sdk()
_driver_mod = importlib.import_module("ao_shaping.drivers.ccd.daheng.driver")
if _driver_mod.gx is None:
    importlib.reload(_driver_mod)
