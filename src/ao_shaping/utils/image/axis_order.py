"""2D 坐标轴序的单一事实源: 项目点序是 ``(x, y)``, 库原生序是 ``(row, col)``。

为什么需要这个模块
------------------
「光斑中心 / 零级坐标」在 ``src/`` 里有两种声明序, 而且都以裸
``tuple[int, int]`` 流转, 约定只写在 docstring 里 —— 没有任何工具能检查:

- **项目点序 ``(x, y)``** (x = 列, x 在前): 占绝对多数, 是运行期约定。
  ``zero_order_center`` / ``smart_zero_order_center`` / ``median_zero_order_center`` /
  ``windowed_centroid`` / ``measure_spot`` / ``clamp_center_to_frame`` 全在此列。
- **row 先序 ``(row, col)`` / ``(y, x)`` / ``(cy, cx)``**: numpy / scipy 的原生
  返回序, 以及若干标定与重采样接口沿用该序。清单见
  :data:`ROW_FIRST_SURFACES` (机器校验, 不要在别处再抄一份)。

方帧上两者不可区分: ``(row, col)`` 与 ``(col, row)`` 都落在合法范围内, 转置是
**静默**的。只有非方帧才稳定暴露 —— 实测 1944x2592 的 CCD 上, 把
``zero_order_center`` 的 ``(x, y)`` 喂给 row 先消费的接口会让整幅 ROI 错位。
因此本仓的守卫测试 (``tests/ao_shaping/test_point_axis_convention.py``) 一律用
``(120, 160)`` 这类非方图, 方图断言是无效断言。

用法
----
运行期把点当 :class:`Point` 用 (属性名即序, 解包即 ``(x, y)``);

>>> row, col = 100, 140
>>> p = rowcol_to_xy((row, col))
>>> (p.x, p.y)
(140, 100)
>>> xy_to_rowcol((p.x, p.y))
(100, 140)

新增一条 row 先接口时的三件事 (缺一不可): ①把它登记进
:data:`ROW_FIRST_SURFACES`; ②docstring 写明 ``(row, col)`` / ``(cy, cx)`` /
``(y, x)`` 之一; ③补一个非方帧断言。登记表由 AST 守卫逐条核对 docstring, 所以
「沉默的序」无法长期存在。
"""

from __future__ import annotations

from typing import Final, NamedTuple

#: 项目点序: ``(x, y)``, x = 列 (水平), y = 行 (垂直), x 在前。
AXIS_XY: Final = "x,y"

#: 库原生序: ``(row, col)``, row = 行 (垂直) 在前。``np.unravel_index`` /
#: ``scipy.ndimage.center_of_mass`` / ``np.argwhere`` 的返回值都属此序。
AXIS_ROWCOL: Final = "row,col"


class Point(NamedTuple):
    """运行期的 2D 点: **x 在前**, 字段名自带轴信息。

    用属性 (``p.x`` / ``p.y``) 而不是位置解包, 是因为 ``cx, cy = ...`` 这种写法
    在方帧上写反了也能跑。库原生 row 先的返回值必须在边界处经
    :func:`rowcol_to_xy` 转换后再构造本类型, 不允许把 row 先元组直接当 ``Point``。
    """

    x: int
    y: int


def rowcol_to_xy(pair: tuple[float, float]) -> Point:
    """把库原生的 ``(row, col)`` 转成项目点序 :class:`Point` (x 在前)。"""
    row, col = pair
    return Point(int(col), int(row))


def xy_to_rowcol(pair: tuple[float, float]) -> tuple[int, int]:
    """把项目点序 ``(x, y)`` 转成库原生 ``(row, col)`` (给索引/切片用)。"""
    x, y = pair
    return int(y), int(x)


#: row 先的公共接口清单 —— **本仓唯一的一份**。
#:
#: 每项是 ``(仓库相对路径, 函数名)``。守卫测试对它逐条做两件事: 用 AST 解析文件,
#: 断言该函数的 docstring 明确写出了 row 先序 (``(row, col)`` / ``(cy, cx)`` /
#: ``(y, x)`` 之一); 对纯函数另外跑一次非方帧行为断言。
#:
#: 这份表存在的意义不是文档, 而是**准入清单**: 想新加一个 row 先的生产者或参数,
#: 只能改这里 —— 于是它出现在 review diff 里, 而不是像现在这样悄悄诞生。
#: 注意 ``hadamard_calc.generate_row_col`` 曾被审计列为 row 先, 但它接收的是
#: Walsh 序数 ``(u, v)``、返回的是相位数组, **不是坐标对**, 故不登记。
ROW_FIRST_SURFACES: Final[tuple[tuple[str, str], ...]] = (
    # 帧内峰值/零级生产者: 直接返回 np.unravel_index 的 row 先结果。
    ("src/ao_shaping/utils/image/beam_metrics.py", "locate_peak"),
    ("src/ao_shaping/optimizer/wfless/_measure.py", "locate_zero_order"),
    # 标定: 面板光束中心与 CCD 零级中心, 写进 calib 工件后仍是 row 先。
    ("src/ao_shaping/tools/slm/calibration/calibration.py", "find_beam_on_slm"),
    (
        "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py",
        "_calib_center",
    ),
    (
        "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/measure.py",
        "_locate_zero_order",
    ),
    # 衍射效率: 零级中心作为 row 先入参穿过整条扫描管线。
    (
        "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/core.py",
        "measure_orders",
    ),
    (
        "src/ao_shaping/tools/slm/calibration/diffraction_efficiency/core.py",
        "run_sweep",
    ),
    # 图样峰值: argwhere 结果原样返回 (y, x)。
    (
        "src/ao_shaping/tools/slm/cartographer/cosine_pattern.py",
        "get_pattern_peak_position",
    ),
    # 重采样: 裁剪中心以 (cy, cx) 传入。
    ("src/ao_shaping/utils/image/resample.py", "crop_to_aspect"),
    ("src/ao_shaping/utils/image/resample.py", "resample_to_grid"),
    # 目标掩膜: 方形支撑以 (cy, cx) 为中心。
    ("src/ao_shaping/utils/image/targets.py", "square_mask"),
    # 全频随机相位整形: 目标图样以 (cy, cx) 为中心栅格化。
    ("src/ao_shaping/optimizer/wfless/slm_gs_refine.py", "_rasterise_target"),
    # 数字孪生: 零级质心按 (row, col) 返回。
    ("src/ml/sim2f/bench.py", "_zero_order_centroid"),
)
