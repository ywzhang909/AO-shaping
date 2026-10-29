"""纯模块半：``sampling_assist`` 的字段、默认、换算与越界建议。

与 ``test_link_design_assist.py`` 同一风格：只测纯函数，不碰 streamlit / 设备。
"""

from __future__ import annotations

import math

import pytest

from ao_shaping.gui.slm import sampling_assist as sa
from ao_shaping.utils.wavefront import sampling_limit as sl


def test_fields_have_explicit_si_name() -> None:
    # 不能靠 removesuffix 猜：main_side_mm 不是光路参数，si_name 必须显式为空
    main = next(f for f in sa.SAMPLING_FIELDS if f.name == "main_side_mm")
    assert main.si_name == ""
    assert not main.is_optical
    assert main.default == sa.DEFAULT_TARGET_SIDE_MM
    for f in sa.SAMPLING_FIELDS:
        if f.name != "main_side_mm":
            assert f.is_optical, f.name


def test_defaults_from_sampling_limit() -> None:
    # 光路参数默认值必须与 DEFAULT_SAMPLING 同源，避免 popover 抄一份
    assert sa.SAMPLING_FIELDS[1].default == pytest.approx(125.0)  # f_main_mm
    assert sa.SAMPLING_FIELDS[2].default == pytest.approx(50.0)  # f_sample_mm
    assert sa.SAMPLING_FIELDS[3].default == pytest.approx(4.0)  # expansion
    assert sa.SAMPLING_FIELDS[4].default == pytest.approx(1.0)  # waist_mm


def test_build_setup_unknown_key_raises() -> None:
    with pytest.raises(TypeError):
        sa.build_setup(unknown=1.0, camera_pixel_m=2.2e-6)


def test_build_setup_missing_camera_raises() -> None:
    with pytest.raises(TypeError):
        sa.build_setup(main_side_mm=40.0)


def test_worked_example() -> None:
    s = sa.build_setup(
        main_side_mm=40.0, f_main_mm=1000.0, f_sample_mm=50.0,
        expansion=4.0, waist_mm=1.0, camera_pixel_m=2.2e-6,
    )
    g = sl.evaluate_sampling(s)
    sm = sa.compute_sampling_summary(g, window_side_px=250)
    assert sm["side_on_camera_mm"] == pytest.approx(2.0)
    assert sm["side_on_camera_px"] == pytest.approx(909.0909, rel=1e-3)
    assert sm["beam_diameter_mm"] == pytest.approx(8.0)
    assert sm["main_resolution_um"] == pytest.approx(133.0)
    assert sm["sampling_ratio"] == pytest.approx(0.83125, rel=1e-3)
    assert sm["fits_window"] is False
    assert sm["ok"] is False


def test_window_fit() -> None:
    # 读出窗 1000 px 装得下 909 px
    s = sa.build_setup(
        main_side_mm=40.0, f_main_mm=1000.0, f_sample_mm=50.0,
        expansion=4.0, waist_mm=1.0, camera_pixel_m=2.2e-6,
    )
    sm = sa.compute_sampling_summary(sl.evaluate_sampling(s), window_side_px=1000)
    assert sm["fits_window"] is True


def test_violation_advice_mapping() -> None:
    s = sa.build_setup(
        main_side_mm=40.0, f_main_mm=1000.0, f_sample_mm=50.0,
        expansion=4.0, waist_mm=1.0, camera_pixel_m=2.2e-6,
    )
    sm = sa.compute_sampling_summary(sl.evaluate_sampling(s), window_side_px=250)
    rows = sa.violation_rows(sm)
    labels = [r[0] for r in rows]
    assert any("欠采样" in l for l in labels)
    assert any("面板物理尺寸" in l for l in labels)


def test_format_length_mm() -> None:
    assert sa.format_length_mm(2.0e-3) == "2.00 mm"
    assert sa.format_length_mm(500e-6) == "500 µm"


def test_summary_rows_has_fit_row() -> None:
    s = sa.build_setup(
        main_side_mm=40.0, f_main_mm=1000.0, f_sample_mm=50.0,
        expansion=4.0, waist_mm=1.0, camera_pixel_m=2.2e-6,
    )
    sm = sa.compute_sampling_summary(sl.evaluate_sampling(s), window_side_px=250)
    rows = sa.summary_rows(sm)
    assert any("读出窗" in r[0] for r in rows)


def test_summary_rows_without_window() -> None:
    s = sa.build_setup(
        main_side_mm=40.0, f_main_mm=1000.0, f_sample_mm=50.0,
        expansion=4.0, waist_mm=1.0, camera_pixel_m=2.2e-6,
    )
    sm = sa.compute_sampling_summary(sl.evaluate_sampling(s))
    rows = sa.summary_rows(sm)
    assert not any("读出窗" in r[0] for r in rows)
