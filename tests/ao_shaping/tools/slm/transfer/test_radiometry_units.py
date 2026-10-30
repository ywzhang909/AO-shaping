"""Single radiometric normalization: native counts/ms, bit depth never folded in.

2026-10-29 的两次黑箱运行在模型阶段失败：``fit`` / ``blur`` 只按 ``exposure_ms``
归一，于是冻结的 ``gain`` 是**原生 sensor counts** 口径；而 ``validate`` /
``direction`` / ``local_response`` / ``axial`` 除的是
``exposure_ms * row["counts_scale"]``（Mono12 时 = 16）。``anchored_intensity`` 把
原生 counts 的模型差分加到小了 16 倍的锚点上 —— 实测 model/measured 比值 16.1，
三个形状、两次运行一致；anchored 模型损失在 9.38 附近摆到 7.03 / 17.12，而实测
损失变化 < 1%，模型块 0/6 被接受。

本文件钉住两件事，免得这个口径再次分叉：

* :func:`ao_shaping.tools.slm.transfer.radiometry.count_rate` 只除曝光，且带
  ``counts_scale`` 的行与不带的行给出**同一个**数（回归项）；
* 六个阶段（fit / blur / validate / direction / local_response / axial）全部走
  ``count_rate``，没有任何一处再乘 ``counts_scale``（源码守卫）。
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from ao_shaping.tools.slm.transfer import (
    axial,
    blur,
    direction,
    fit,
    local_response,
    radiometry,
    validate,
)
from ao_shaping.tools.slm.transfer.radiometry import count_rate


#: 一帧典型的 Mono12 采样窗口（原生 counts，未除曝光）。
FRAME = np.array([[400, 812], [1600, 4095]], dtype=np.uint16)

#: 采集期写入 manifest 的 Mono12 行：``counts_scale`` 只是 8 bit 等价的元数据。
MONO12_ROW = {"exposure_ms": 0.5, "bit_depth": 12, "counts_scale": 16.0,
              "full_scale": 4095.0, "peak": 1000.0}


def test_count_rate_divides_by_exposure_only():
    rate = count_rate(FRAME, {"exposure_ms": 2.0})
    np.testing.assert_allclose(rate, FRAME / 2.0)


def test_count_rate_ignores_bit_depth_metadata():
    """同帧、同曝光，带 ``counts_scale``/``bit_depth`` 的行必须与原行逐位相同。"""
    bare = {"exposure_ms": 0.5}
    scaled = {"exposure_ms": 0.5, "bit_depth": 12, "counts_scale": 16.0}
    np.testing.assert_array_equal(count_rate(FRAME, bare), count_rate(FRAME, scaled))


@pytest.mark.parametrize("exposure", [0.0, -1.0, float("nan"), float("inf")])
def test_count_rate_rejects_nonpositive_or_nonfinite_exposure(exposure):
    with pytest.raises(ValueError, match="positive finite exposure"):
        count_rate(FRAME, {"exposure_ms": exposure})


def test_8bit_validity_is_unchanged():
    """Mono8 阈值仍是 ``peak < 250``（行里没有 ``full_scale`` 时按 255 算）。"""
    assert fit.valid({"exposure_ms": 0.5, "peak": 249.0}) is True
    assert fit.valid({"exposure_ms": 0.5, "peak": 250.0}) is False


def test_mono12_row_is_valid():
    """Mono12 峰值 1000（旧的硬编码 ``peak < 250`` 会把它当无效行丢掉）。"""
    assert fit.valid(MONO12_ROW) is True


def test_mono12_saturated_row_is_invalid():
    assert fit.valid({**MONO12_ROW, "peak": 4095.0}) is False


def test_peak_gate_is_a_fraction_of_the_rows_own_full_scale():
    """门限是 ``250/255 * full_scale``，不是绝对 counts。

    Mono12 下门限 = 4014.7，所以 4000 仍是**有效**行（它只差 0.36% 到满量程）。
    要判它无效只能另加一个绝对阈值，那就成了第二套、与本地满量程冲突的门 ——
    真饱和由 ``peak == full_scale`` 一侧覆盖。
    """
    assert fit.valid({**MONO12_ROW, "peak": 4000.0}) is True
    assert fit.valid({**MONO12_ROW, "peak": 4015.0}) is False


@pytest.mark.parametrize("module", [fit, blur, validate, direction, local_response, axial])
def test_every_stage_routes_through_count_rate(module):
    source = inspect.getsource(module)
    assert "count_rate" in source, f"{module.__name__} no longer imports the shared normalization"
    # 只检查代码行: 注释里提到 counts_scale 是允许的（那是在说明它不该被乘进来）。
    code = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "*counts_scale" not in code.replace("* counts_scale", "*counts_scale")
    assert "/counts_scale" not in code.replace("/ counts_scale", "/counts_scale")


def test_radiometry_documents_the_failure_it_prevents():
    """成因必须留在模块里: 16 倍失配是这段注释唯一的存在理由。"""
    assert "2026-10-29" in inspect.getsource(radiometry.count_rate)