"""上游 OOPAO 0.x 的兼容垫片。

上游的 ``OOPAO/__init__.py`` 在 numpy 2.x + Windows 上有两处坏掉:

1. 它在 ``sys.path`` 中搜索包含 ``"OOPAO"`` 的条目以定位自身的安装目录, 并往里写
   ``precision_oopao.npy``。当包被正常安装时没有任何 ``sys.path`` 条目能匹配上, 该查找
   会抛出 ``ValueError: attempt to get argmin of an empty sequence``。
2. 就算过了上一关, 它还会执行 ``from OOPAO.tools.tools import warning`` —— 而这个
   子模块在上游 OOPAO 里已经不再随包发布 (``tools/`` 目录在模块被平铺进顶层包时删掉了)。
   因此导入 ``OOPAO`` 包总是失败。

变通做法: 根本不去 import ``OOPAO`` 这个包。而是用 ``importlib`` 直接从已安装的包目录
加载我们用到的那几个模块, 并在 ``sys.modules`` 里植入一个**影子** ``OOPAO`` 包对象
(只有 ``__path__``), 使这些模块内部的相对导入 (``phaseStats`` → ``tools.tools``)
能够解析, 而完全不必执行那份坏掉的 ``OOPAO/__init__.py``。上面那些上游 bug 被彻底绕开。

整套件访问 (``calibration``、``closed_loop``、
``mis_registration_identification_algorithm``) 靠把源码 checkout 的包父目录追加到影子
``__path__`` 实现: 上游 ``setup.cfg`` 只列了 ``packages = OOPAO``, 所以 pip wheel 里
不含这些子包, 但 pin 住的源码克隆 (同一 commit ``8e12a17f``) 里是有的, 于是通过影子包解析。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType

import numpy as np

_PKG_NAME = "OOPAO"
# 我们会主动加载的全部顶层模块 (SPRINT 是按需加载的; 它 import 的
# ``OOPAO.calibration`` 可通过追加的源码路径拿到)。
_MODULES = (
    "Asterism", "Atmosphere", "BioEdge", "DeformableMirror", "Detector",
    "FieldTransformer", "GainSensingCamera", "InfluenceFunctions", "LiFT",
    "MisRegistration", "NCPA", "OPD_map", "Pyramid", "ShackHartmann",
    "SpatialFilter", "Source", "Telescope", "Zernike", "phaseStats",
)

_spec = importlib.util.find_spec(_PKG_NAME)
if _spec is None or _spec.origin is None:
    raise ImportError("OOPAO package not importable; install via `uv sync`.")
_pkg_dir = os.path.dirname(_spec.origin)

# =====================================================================
# 上游变通处理 (numpy 2.x / Windows)
# =====================================================================
# 1) OOPAO/__init__.py 与 Telescope.__init__ 都是靠扫描 ``sys.path`` 里包含 "OOPAO"
#    的条目、再对匹配的路径长度做 ``np.argmin`` 来定位安装目录。正常安装时没有任何
#    条目能匹配上, 于是抛出 ``ValueError: attempt to get argmin of an
#    empty sequence``。修法: 让安装目录在 ``sys.path`` 上可见。
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

# 2) 接着 Telescope.__init__ 会加载 ``<pkg_dir>/precision_oopao.npy`` 来选择
#    float64/float32 精度。上游是由那份 (坏掉且现在被跳过的) 包 __init__ 写这个文件的,
#    所以缺失时在此创建。
_precision_file = os.path.join(_pkg_dir, "precision_oopao.npy")
if not os.path.exists(_precision_file):
    try:
        np.save(_precision_file, 64)
    except OSError:
        pass  # site-packages 只读: 下面回落到 float64 默认值

# 把 ``sys.modules`` 里的包条目替换成一个轻量的*影子*包对象 (一个只带 ``__path__``
# 的普通 ModuleType)。我们从不执行上游的 ``OOPAO/__init__.py`` —— 它那套坏掉的
# ``sys.path`` 搜索加上缺失的 ``tools`` 导入在 numpy 2.x/Windows 上必崩 —— 而子模块
# 内部的相对导入 (如 ``phaseStats`` 的 ``from .tools.tools import ...``) 会
# *穿过*这个影子父包解析, 而不是再次触发那份坏掉的包初始化。
if _PKG_NAME in sys.modules:
    sys.modules.pop(_PKG_NAME, None)
_shadow = ModuleType(_PKG_NAME)
_shadow.__path__ = [_pkg_dir]
sys.modules[_PKG_NAME] = _shadow

# 3) ``setup.cfg`` 只声明了 ``packages = OOPAO``, 因此装出来的 wheel 会漏掉
#    ``calibration/``、``closed_loop/`` 与
#    ``mis_registration_identification_algorithm/`` 这些子包, 尽管它们在源码树里是
#    存在的。把源码的包父目录追加到影子 ``__path__``, 使 ``from
#    OOPAO.calibration.InteractionMatrix import ...`` 能通过那份 (pin 住的、
#    同一 commit 的) 源码 checkout 解析。克隆不存在时静默回落。
#    (可选的额外路径; 核心那 18 个模块仅靠 ``_pkg_dir`` 就能加载。)
_SOURCE_PKG_DIR = os.environ.get("AO_OOPAO_SOURCE_DIR", "")
if _SOURCE_PKG_DIR and os.path.isdir(_SOURCE_PKG_DIR) and _SOURCE_PKG_DIR not in _shadow.__path__:
    _shadow.__path__.append(_SOURCE_PKG_DIR)


def _load_submodule(name: str) -> ModuleType:
    """加载一个启用根相对导入的 OOPAO 顶层模块。"""
    path = os.path.join(_pkg_dir, f"{name}.py")
    spec = importlib.util.spec_from_file_location(
        f"{_PKG_NAME}.{name}", path, submodule_search_locations=[_pkg_dir]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OOPAO submodule {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    # 根相对的 ``from .X import Y`` 必须相对 OOPAO 包解析 (如 SPRINT 的
    # ``from .calibration.CalibrationVault import``); 不设这里的话, 相对导入会锚定在
    # ``OOPAO.<name>`` 上。
    mod.__package__ = _PKG_NAME
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_loaded = {name: _load_submodule(name) for name in _MODULES}

Atmosphere = _loaded["Atmosphere"].Atmosphere
Source = _loaded["Source"].Source
Telescope = _loaded["Telescope"].Telescope
DeformableMirror = _loaded["DeformableMirror"].DeformableMirror
Zernike = _loaded["Zernike"].Zernike
Asterism = _loaded["Asterism"].Asterism
Detector = _loaded["Detector"].Detector
Pyramid = _loaded["Pyramid"].Pyramid
ShackHartmann = _loaded["ShackHartmann"].ShackHartmann
FieldTransformer = _loaded["FieldTransformer"].FieldTransformer
SpatialFilter = _loaded["SpatialFilter"].SpatialFilter
NCPA = _loaded["NCPA"].NCPA
OPD_map = _loaded["OPD_map"].OPD_map
phaseStats = _loaded["phaseStats"]
InfluenceFunctions = _loaded["InfluenceFunctions"]
BioEdge = _loaded["BioEdge"].BioEdge
GainSensingCamera = _loaded["GainSensingCamera"].GainSensingCamera
LiFT = _loaded["LiFT"].LiFT
MisRegistration = _loaded["MisRegistration"].MisRegistration

# 经追加的源码路径解析的子包:
#   OOPAO.calibration.InteractionMatrix / compute_KL_modal_basis /
#   CalibrationVault / get_modal_basis / initialization_AO* ...
#   OOPAO.closed_loop.run_cl* ...
#   OOPAO.mis_registration_identification_algorithm.* ...
__all__ = [
    "Atmosphere", "Source", "Telescope", "DeformableMirror", "Zernike",
    "Asterism", "Detector", "Pyramid", "ShackHartmann", "FieldTransformer",
    "SpatialFilter", "NCPA", "OPD_map", "phaseStats", "InfluenceFunctions",
    "BioEdge", "GainSensingCamera", "LiFT", "MisRegistration",
]
