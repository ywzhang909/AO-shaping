"""回归: 导入相机注册表**不得**加载原生相机 SDK (PEP 562 惰性后端)。

回归背景 (2026-09-29): ``drivers/__init__.py`` 与 ``drivers/ccd/__init__.py``
原先在包导入期就 eager import MiiCam 与 Daheng 后端。Python 会先初始化父包
再导入其子模块, 因此**任何** ``ao_shaping.drivers.ccd.common`` 的使用者都会
被迫承担两个副作用:

* ``miicam/driver.py`` 在模块作用域执行 ``_setup_miicam_sdk()``, 该函数会改写
  ``sys.path`` 并用 ``ctypes.CDLL`` 预加载原生 ``MIIUSB.dll``;
* ``daheng/driver.py`` 导入 ``gxipy`` 绑定。

最典型的受害者是离线报告脚本 ``scripts/generate_shape_objective_comparison.py``
—— 它只想拿到纯函数 ``square_quality_score``, 却打印出
``MIICAM SDK set up successfully`` 并触发 gxipy 缺失报错。

现在两个 SDK 后端改为通过 PEP 562 模块级 ``__getattr__`` 惰性解析, 仅在首次
属性访问时才导入, 与 ``ccd/AGENTS.md`` 中对 ``create_camera`` 注册表
"某个后端 SDK 缺失时仅在该类型被请求时失败" 的既有约定一致。

注意: 断言必须跑在**子进程**中。pytest 收集到本模块时, 早先的测试通常已经把
SDK 导入进 ``sys.modules``, 进程内检查将失去意义。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SDK_MODULES = ("miicam", "gxipy")

# 离线报告脚本只需纯函数, 不应触发任何原生 SDK。
_OFFLINE_CONSUMERS = (
    "ao_shaping.drivers.ccd.common",
    "scripts.generate_shape_objective_comparison",
)


def _run_snippet(snippet: str) -> subprocess.CompletedProcess[str]:
    """在干净解释器中执行 ``snippet`` 并返回结果。

    显式拼装 ``PYTHONPATH`` (``src`` + ``libs`` + 继承值), 复现 ``.env`` 中
    的运行环境 —— 缺了 ``libs`` 时 ``gxipy`` 不可导入, 会让本测试的
    "已加载 SDK" 断言产生误判。
    """
    env = dict(os.environ)
    parts = [str(_REPO_ROOT / "src"), str(_REPO_ROOT / "libs")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)

    return subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("module", _OFFLINE_CONSUMERS)
def test_offline_import_does_not_load_native_sdks(module: str) -> None:
    """离线消费者导入后, ``miicam`` / ``gxipy`` 都不应出现在 ``sys.modules``。"""
    probe = ", ".join(repr(name) for name in _SDK_MODULES)
    snippet = (
        f"import sys\n"
        f"import {module}\n"
        f"loaded = [n for n in ({probe},) if n in sys.modules]\n"
        f"print(','.join(loaded))\n"
    )

    result = _run_snippet(snippet)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"importing {module} eagerly loaded native SDK module(s): "
        f"{result.stdout.strip()}"
    )


def test_public_camera_api_still_resolves() -> None:
    """惰性化不得改变公共 API: 原 ``from ao_shaping.drivers import MIICamera`` 仍可用。"""
    snippet = (
        "from ao_shaping.drivers import MIICamera, DahengCamera\n"
        "from ao_shaping.drivers.ccd import (MIICamera as C1, MIICAMError,\n"
        "    DahengCamera as C2, FFmpegCamera, ImageFolderCamera)\n"
        "assert C1 is MIICamera and C2 is DahengCamera\n"
        "assert MIICAMError is not None\n"
        "assert FFmpegCamera.__name__ == 'FFmpegCamera'\n"
        "assert ImageFolderCamera.__name__ == 'ImageFolderCamera'\n"
        "print('ok')\n"
    )

    result = _run_snippet(snippet)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_camera_registry_still_lists_all_backends() -> None:
    """惰性化后注册表内容不变 —— 惰性的是 *导入*, 不是 *可用性*。"""
    snippet = (
        "from ao_shaping.drivers.ccd.common import list_camera_types\n"
        "print(','.join(list_camera_types()))\n"
    )

    result = _run_snippet(snippet)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "daheng,ffmpeg,image_folder,miicam,sim"


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """``__getattr__`` 不得吞掉真正的拼写错误。"""
    snippet = (
        "import ao_shaping.drivers as d\n"
        "import ao_shaping.drivers.ccd as c\n"
        "for mod in (d, c):\n"
        "    try:\n"
        "        mod.NoSuchCamera\n"
        "    except AttributeError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('expected AttributeError')\n"
        "print('ok')\n"
    )

    result = _run_snippet(snippet)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
