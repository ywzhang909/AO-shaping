"""R-64 §8.9 — WFS 曝光安全上限: 单一归属 + 显式 raise (不用 assert)。

三个守卫, 全部离线 (不开任何设备):

* `require_wfs_exposure_within_safety_ceiling` 的行为契约 —— 上限值原样返回,
  超限抛 **ValueError 且显式不是 AssertionError**。这是 `python -O` 存活证明:
  `assert` 在 -O 下被整条移除, 而硬件读回的曝光是外部状态, 联锁必须在两种
  解释器模式下都生效。
* `drivers/ccd/miicam` 的 `MIICamera.MAX_EXPOSURE_MS = 10000.0` 是**另一个量**
  (相机器件级上限, 比 WFS 激光安全上限大 ~1400 倍), 必须保持原值 —— 旧名
  `MAX_EXPOSURE_MS` 在两个语义上冲突, 这正是改名的理由。
* AST 守卫: `src/ao_shaping/tools/` 与 `scripts/` 下没有任何 `Assert` 节点引用
  `WFS_EXPOSURE_SAFETY_MAX_MS`; 该常量在模块级**只被赋值一次** (单一归属)。
  扫描风格沿用 `tests/ao_shaping/scripts/test_common_helpers_not_reintroduced.py`,
  读文件用 `encoding="utf-8-sig"` (一个已入库脚本带 BOM)。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ao_shaping.tools.zernike.common import (
    WFS_EXPOSURE_SAFETY_MAX_MS,
    require_wfs_exposure_within_safety_ceiling,
)

_ROOT = Path(__file__).resolve().parents[4]
_TOOLS = _ROOT / "src" / "ao_shaping" / "tools"
_SCRIPTS = _ROOT / "scripts"
_SRC = _ROOT / "src"
_MIICAM_DRIVER = _ROOT / "src" / "ao_shaping" / "drivers" / "ccd" / "miicam" / "driver.py"

_CEILING_NAME = "WFS_EXPOSURE_SAFETY_MAX_MS"


def _read(path: Path) -> str:
    """Read a source file, tolerating a UTF-8 BOM (one checked-in script has one)."""
    return path.read_text(encoding="utf-8-sig")


def _py_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if "__pycache__" not in p.parts and p.is_file()
    )


# ---------------------------------------------------------------------------
# 1. helper 行为契约 + -O 存活证明
# ---------------------------------------------------------------------------

def test_value_at_ceiling_returned_unchanged() -> None:
    """恰好等于上限的曝光必须原样返回 (上限是闭区间, 与 CLI 的 `>` 判定一致)。"""
    assert require_wfs_exposure_within_safety_ceiling(
        WFS_EXPOSURE_SAFETY_MAX_MS
    ) == WFS_EXPOSURE_SAFETY_MAX_MS


def test_value_above_ceiling_raises_value_error_not_assertion_error() -> None:
    """超限必须抛 ValueError, 且显式不是 AssertionError。

    这就是 `python -O` 存活证明: 联锁走的是普通 `raise`, 优化等级不剥离它;
    而 `assert` 在 -O 下整条消失, 对硬件读回的外部状态等于没有联锁。
    """
    with pytest.raises(ValueError) as excinfo:
        require_wfs_exposure_within_safety_ceiling(
            WFS_EXPOSURE_SAFETY_MAX_MS + 0.001
        )
    exc = excinfo.value
    assert not issubclass(type(exc), AssertionError), (
        "安全联锁必须是显式 raise 的 ValueError, 不是 assert —— "
        "`python -O` 会整条移除 assert"
    )
    assert type(exc) is ValueError


def test_helper_returns_float_and_accepts_int_like_input() -> None:
    """helper 把输入转成 float 后返回 (与 `exp = float(wfs.exposure_time)` 语义一致)。"""
    assert require_wfs_exposure_within_safety_ceiling(3) == 3.0


# ---------------------------------------------------------------------------
# 2. MIICamera 器件级上限是另一个量, 名字不得冲突
# ---------------------------------------------------------------------------

def _miicam_class_assignments() -> dict[str, ast.expr]:
    tree = ast.parse(_read(_MIICAM_DRIVER), filename=str(_MIICAM_DRIVER))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MIICamera":
            return {
                t.id: node_stmt.value
                for node_stmt in node.body
                if isinstance(node_stmt, ast.Assign)
                for t in node_stmt.targets
                if isinstance(t, ast.Name)
            }
    raise AssertionError("MIICamera class not found in miicam/driver.py")


def test_miicam_device_max_exposure_is_a_different_quantity() -> None:
    """`MIICamera.MAX_EXPOSURE_MS` 是相机器件级上限, 必须保持 10000.0。

    它与 WFS 激光安全上限 (7.0 ms) 相差 ~1400 倍, 是两个完全不同的量 ——
    旧名 `MAX_EXPOSURE_MS` 让二者在搜索/阅读时相互冒充, 这正是本次改名把
    WFS 侧收敛为 `WFS_EXPOSURE_SAFETY_MAX_MS` 的理由。driver 侧逐字节不动。
    """
    assignments = _miicam_class_assignments()
    device_max = assignments.get("MAX_EXPOSURE_MS")
    assert isinstance(device_max, ast.Constant), (
        "MIICamera.MAX_EXPOSURE_MS 的声明被改动; 它是器件级上限, 不在本次收口范围"
    )
    assert device_max.value == 10000.0
    assert device_max.value != WFS_EXPOSURE_SAFETY_MAX_MS, (
        "两个上限相等说明名字又撞了: 器件级 10000.0 ms 与激光安全 7.0 ms 是两个量"
    )


# ---------------------------------------------------------------------------
# 3. AST 守卫
# ---------------------------------------------------------------------------

def _names_in_node(node: ast.AST) -> set[str]:
    return {
        n.id for n in ast.walk(node) if isinstance(n, ast.Name)
    } | {
        n.value for n in ast.walk(node)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }


def test_no_assert_node_references_the_ceiling_constant() -> None:
    """`src/ao_shaping/tools/` 与 `scripts/` 下禁止 `assert ... WFS_EXPOSURE_SAFETY_MAX_MS`。

    曝光上限守的是硬件读回的外部状态, 全部 6 处已收口为
    `require_wfs_exposure_within_safety_ceiling`; 任何回归成 `assert` 的写法
    都会在 `python -O` 下静默失效, 这里在 AST 层直接拦掉。
    """
    offenders: list[str] = []
    for root in (_TOOLS, _SCRIPTS):
        for path in _py_files(root):
            tree = ast.parse(_read(path), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert) and _CEILING_NAME in _names_in_node(node):
                    offenders.append(
                        f"{path.relative_to(_ROOT).as_posix()}:{node.lineno}"
                    )
    assert not offenders, (
        f"这些位置把安全联锁写回了 assert ({offenders}); "
        "改用 require_wfs_exposure_within_safety_ceiling()"
    )


def test_ceiling_constant_has_a_single_module_level_home() -> None:
    """`WFS_EXPOSURE_SAFETY_MAX_MS` 在 `src/` 与 `scripts/` 里只有一处模块级赋值。

    此前 `MAX_EXPOSURE_MS = 7.0` 在 3 个文件里各写一份 (tools/wfs 两处 +
    tools/slm/calibration 一处), 现在唯一归属是
    `src/ao_shaping/tools/zernike/common.py`。
    """
    homes: list[str] = []
    for root in (_SRC, _SCRIPTS):
        for path in _py_files(root):
            tree = ast.parse(_read(path), filename=str(path))
            for node in tree.body:  # 模块级 only
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == _CEILING_NAME
                    for t in node.targets
                ):
                    homes.append(path.relative_to(_ROOT).as_posix())
                elif isinstance(node, ast.AnnAssign) and isinstance(
                    node.target, ast.Name
                ) and node.target.id == _CEILING_NAME:
                    homes.append(path.relative_to(_ROOT).as_posix())
    assert homes == ["src/ao_shaping/tools/zernike/common.py"], (
        f"{_CEILING_NAME} 的模块级赋值出现在 {homes}; 唯一归属必须是 "
        "src/ao_shaping/tools/zernike/common.py"
    )
