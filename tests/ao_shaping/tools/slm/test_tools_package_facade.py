"""``ao_shaping.tools.slm`` / ``ao_shaping.tools.micro_dm`` 包级 facade 契约。

锁定两件事 (TODO.md R-30 / R-40):

1. **包级不做 eager re-export**。子模块在模块作用域 import ``click`` / ``loguru``
   / numpy / 相机与 SLM 驱动, 所以包级再导出会让 *任何*
   ``import ao_shaping.tools.slm.<x>`` 付出整套探针模块的导入代价。
   ``ao_shaping.tools.slm.__all__`` 必须为空, 且模块源码里不得出现
   ``from ao_shaping...`` / ``import ao_shaping...`` 顶层语句。
2. **``__init__.py`` 必须存在**。缺 ``__init__.py`` 的目录会被 ``find_packages()``
   式打包**静默丢弃** (TODO.md R-24)。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[4] / "src" / "ao_shaping" / "tools"
_SLM_PKG = _PKG_ROOT / "slm"
_MICRO_DM_PKG = _PKG_ROOT / "micro_dm"


def _top_level_imports(path: Path) -> list[ast.Import | ast.ImportFrom]:
    """Return module-scope ``import`` / ``from ... import`` nodes."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


class TestToolsSlmPackage:
    def test_init_exists(self) -> None:
        assert (_SLM_PKG / "__init__.py").is_file()

    def test_all_is_empty(self) -> None:
        import ao_shaping.tools.slm as pkg

        assert pkg.__all__ == []

    def test_init_has_no_eager_imports(self) -> None:
        init = _SLM_PKG / "__init__.py"
        assert _top_level_imports(init) == [], (
            "ao_shaping.tools.slm.__init__ must not eagerly import submodules; "
            "import the symbol from its own submodule instead"
        )

    @pytest.mark.parametrize("module_name", sorted(p.name for p in _SLM_PKG.glob("*.py")))
    def test_no_facade_star_import_anywhere(self, module_name: str) -> None:
        """``from ao_shaping.tools.slm import *`` would rely on a facade surface."""
        source = (_SLM_PKG / module_name).read_text(encoding="utf-8")
        assert "from ao_shaping.tools.slm import *" not in source


class TestToolsMicroDmPackage:
    def test_init_exists(self) -> None:
        assert (_MICRO_DM_PKG / "__init__.py").is_file()

    def test_init_has_no_eager_imports(self) -> None:
        init = _MICRO_DM_PKG / "__init__.py"
        assert _top_level_imports(init) == []

    def test_submodule_is_importable(self) -> None:
        """Regression for the 12 docstrings that still said the pre-move path."""
        import importlib

        mod = importlib.import_module(
            "ao_shaping.tools.micro_dm.micro_dm_image_collect"
        )
        assert hasattr(mod, "main")


class TestMicroDmDocstringsAreCurrent:
    def test_no_stale_module_path_in_docstrings(self) -> None:
        source = (_MICRO_DM_PKG / "micro_dm_image_collect.py").read_text(encoding="utf-8")
        assert "ao_shaping.tools.micro_dm_image_collect" not in source
        assert "python -m ao_shaping.tools.micro_dm.micro_dm_image_collect" in source