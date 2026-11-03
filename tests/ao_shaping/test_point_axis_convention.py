"""#223 option 1 守卫: 2D 点的轴序必须有名字、有清单、且能被非方帧抓到。

背景: 「光斑中心 / 零级坐标」在 ``src/`` 内以两种互不相容的序流转
(项目 ``(x, y)`` 与库原生 ``(row, col)``), 两边都是裸 ``tuple[int, int]``,
约定只写在 docstring 里。**方帧上转置不可观测** —— ``(row, col)`` 与
``(col, row)`` 都落在合法范围内, 于是跨模块手递手时静默错位, 直到 1944x2592
这类非方 CCD 上整幅 ROI 偏掉才被发现。

因此本文件里所有行为断言都用非方帧 (``H=120, W=320``), 并且刻意把峰值放在
「转置后越界或落到另一侧」的位置: 方帧断言在这里等于没有断言。

三层守卫:
1. ``axis_order`` 的转换原语: 往返精确, 且 ``Point`` 的属性名自带轴信息。
2. 纯函数的**真实返回序**: 用非方帧断言, 不看 docstring 怎么写。
3. 登记表 ``ROW_FIRST_SURFACES``: 每条都要在文件里存在、docstring 显式写出
   row 先序 (AST 解析, 不 import —— ``src/ml`` 的模块带重依赖), 边界处还要有
   ``# AXIS:`` 行内标注。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np

from ao_shaping.optimizer.wfless._measure import locate_zero_order
from ao_shaping.tools.slm.cartographer.cosine_pattern import get_pattern_peak_position
from ao_shaping.utils.image.axis_order import (
    AXIS_ROWCOL,
    AXIS_XY,
    ROW_FIRST_SURFACES,
    Point,
    rowcol_to_xy,
    xy_to_rowcol,
)
from ao_shaping.utils.image.beam_metrics import locate_peak, zero_order_center
from ao_shaping.utils.image.resample import crop_to_aspect, resample_to_grid
from ao_shaping.utils.image.targets import square_mask

_REPO = Path(__file__).resolve().parents[2]

# 非方帧: 行数与列数差得足够远, 转置后峰值必然越界或落在明显偏离处。
_H, _W = 120, 320
_PEAK_ROW, _PEAK_COL = 60, 240

#: docstring 里的 row 先声明。用正则而不是字面量, 因为三元组也常见
#: (``(cy, cx, peak_adu)``) —— 字面量匹配会把它们判成"没写明"。
_ROW_FIRST_RE = re.compile(
    r"\(\s*(?:row|cy|y)\s*,\s*(?:col|cx|x)\b"  # (row, col) / (cy, cx) / (y, x)
    r"|\[\s*row\s*,\s*col\s*\]"  # [row, col] 工件
    r"|行\s*,\s*列"
)


def _peak_frame() -> np.ndarray:
    """单峰非方帧: 唯一亮像素在 ``(_PEAK_ROW, _PEAK_COL)``。"""
    frame = np.zeros((_H, _W), dtype=np.float64)
    frame[_PEAK_ROW, _PEAK_COL] = 1000.0
    return frame


def test_axis_constants_are_distinct_and_documented() -> None:
    assert AXIS_XY == "x,y"
    assert AXIS_ROWCOL == "row,col"
    assert AXIS_XY != AXIS_ROWCOL


def test_point_is_x_first() -> None:
    """``Point`` 的字段名就是契约: ``p.x`` 是列, ``p.y`` 是行。"""
    point = Point(140, 100)
    assert tuple(point) == (140, 100)
    assert point.x == 140
    assert point.y == 100


def test_rowcol_and_xy_roundtrip_on_a_non_square_frame() -> None:
    """往返必须精确, 且转置后的值会越界 —— 这就是非方帧的判据。"""
    row, col = 100, 140

    point = rowcol_to_xy((row, col))
    assert point == Point(col, row)

    back = xy_to_rowcol((point.x, point.y))
    assert back == (row, col)

    # x 在 [0, W), y 在 [0, H): 若两者互换, 140 仍合法而 100 也合法 => 方帧测不出;
    # 用 H != W 的帧时把 row 取在 W 之外才能让转置显形。
    assert 0 <= point.x < _W
    assert 0 <= point.y < _H


def test_transposed_point_is_detectable_against_frame_bounds() -> None:
    """一个落在 W 之外的行号, 转置后就会伪装成合法列号。"""
    out_of_range_row = _W - 10  # 310: 作为行越界 (H=120), 作为列合法
    assert out_of_range_row >= _H

    point = rowcol_to_xy((out_of_range_row, 5))
    assert point.y == _W - 10  # 仍按行保存, 不会被"修好"成合法值
    assert point.y >= _H, "转置后的行号必须能被帧高拒绝, 而不是被静默接受"


def test_locate_peak_is_row_first() -> None:
    assert locate_peak(_peak_frame()) == (_PEAK_ROW, _PEAK_COL)


def test_locate_zero_order_is_row_first() -> None:
    assert locate_zero_order(_peak_frame()) == (_PEAK_ROW, _PEAK_COL)


def test_zero_order_center_is_x_first() -> None:
    """同一文件内的两种序并存 —— 这是 #223 报告的核心矛盾, 这里钉住 x 先那支。"""
    assert zero_order_center(_peak_frame()) == (_PEAK_COL, _PEAK_ROW)


def test_get_pattern_peak_position_is_row_first() -> None:
    assert get_pattern_peak_position(_peak_frame()) == (_PEAK_ROW, _PEAK_COL)


def test_square_mask_takes_cy_cx() -> None:
    mask = square_mask((_H, _W), _PEAK_ROW, _PEAK_COL, 20)
    assert mask.shape == (_H, _W)
    assert mask[_PEAK_ROW, _PEAK_COL], "中心像素必须在掩膜内"
    assert mask.sum() == 20 * 20
    # cy/cx 若被当作 (x, y), 行坐标变成 240 > H, 掩膜会整个落空。
    assert not mask[_PEAK_ROW - 20, _PEAK_COL], "掩膜被放大到了 40 px"


def test_crop_to_aspect_centres_on_the_cx_it_was_given() -> None:
    """中心若被当作 ``(x, y)`` 传入, 裁剪窗口会水平偏 40 px。"""
    cropped = crop_to_aspect(_peak_frame(), 1.0, center=(_PEAK_ROW, _PEAK_COL))
    assert cropped.shape[0] == cropped.shape[1]
    row_idx, col_idx = np.unravel_index(int(np.argmax(cropped)), cropped.shape)
    assert row_idx == _PEAK_ROW
    assert col_idx == cropped.shape[1] // 2, "峰应落在裁剪窗的水平中心"


def test_resample_to_grid_keeps_the_peak_centred() -> None:
    target = (60, 80)
    resampled = resample_to_grid(_peak_frame(), target, center=(_PEAK_ROW, _PEAK_COL))
    assert resampled.shape == target
    row_idx, col_idx = np.unravel_index(int(np.argmax(resampled)), resampled.shape)
    assert abs(row_idx - target[0] // 2) <= 2
    assert abs(col_idx - target[1] // 2) <= 2, "峰应落在输出网格中心 (中心被转置则会跑偏)"


def test_registry_has_no_duplicates_and_only_src_paths() -> None:
    """登记表是准入清单: 空表、重复项、越界路径都说明它已经失去意义。"""
    assert ROW_FIRST_SURFACES, "登记表空了 —— 那条守卫就没有对象可查了"
    assert len(set(ROW_FIRST_SURFACES)) == len(ROW_FIRST_SURFACES)
    for path, _name in ROW_FIRST_SURFACES:
        assert path.startswith("src/"), f"登记表混进了非 src 路径: {path}"


def test_every_registered_surface_exists_and_declares_its_order() -> None:
    """AST 解析而非 import: ``src/ml`` 的模块带重依赖, import 会把守卫拖成慢测试。"""
    problems: list[str] = []
    for path, name in ROW_FIRST_SURFACES:
        file_path = _REPO / path
        if not file_path.is_file():
            problems.append(f"{path}: 文件不存在")
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        found = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        node = found.get(name)
        if node is None:
            problems.append(f"{path}::{name}: 函数不存在 (登记表已过期)")
            continue
        doc = ast.get_docstring(node) or ""
        if not _ROW_FIRST_RE.search(doc):
            problems.append(
                f"{path}::{name}: docstring 没有写明 row 先序 —— 调用方只能靠猜"
            )
    assert not problems, "登记表与代码不一致:\n  " + "\n  ".join(problems)


def test_boundary_functions_carry_an_inline_axis_marker() -> None:
    """跨域边界必须有行内标注: docstring 没人读, ``# AXIS:`` 在 diff 里看得见。"""
    boundaries = {
        ("src/ao_shaping/utils/image/beam_metrics.py", "locate_peak"),
        ("src/ao_shaping/optimizer/wfless/_measure.py", "locate_zero_order"),
        ("src/ao_shaping/tools/slm/cartographer/cosine_pattern.py", "get_pattern_peak_position"),
        ("src/ao_shaping/utils/image/resample.py", "crop_to_aspect"),
        ("src/ao_shaping/utils/image/resample.py", "resample_to_grid"),
    }
    missing: list[str] = []
    for path, name in sorted(boundaries):
        source = (_REPO / path).read_text(encoding="utf-8")
        if f"# AXIS: (row, col)\ndef {name}(" not in source:
            missing.append(f"{path}::{name}")
    assert not missing, "缺少 `# AXIS: (row, col)` 边界标注:\n  " + "\n  ".join(missing)
