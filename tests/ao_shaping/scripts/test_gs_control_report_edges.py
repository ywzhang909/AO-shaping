"""「矫正后发送 GS」那一栏的度量口径 (图 7)。

只测纯函数: 边缘过渡宽度、未整形光斑 FWHM、帧/参数读取、JSON 里的 nan 处理。
这些是 C 与 B 唯一可比的量, 一旦它们算错, 报告里的结论就变成「噪声的解读」,
所以每个都用**已知答案的合成图**钉住, 而不是拿真实帧做回归快照。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from scripts.slm.generate_gs_control_report import (
    _crossing_width,
    _fwhm,
    _num_or_none,
    edge_transition,
    load_frame_with_meta,
    psf_shape,
)


# ---------------------------------------------------------------------------
# 帧 / 参数读取
# ---------------------------------------------------------------------------
def test_load_frame_with_meta_reads_frame_and_params(tmp_path: pathlib.Path) -> None:
    frame = np.arange(12, dtype=np.uint8).reshape(3, 4)
    path = tmp_path / "c.npz"
    np.savez_compressed(path, frame=frame, params=json.dumps({"exposure_ms_readback": 200}))

    got, meta = load_frame_with_meta(path)

    assert got.shape == (3, 4)
    assert got.dtype == np.float64
    assert np.array_equal(got, frame.astype(float))
    assert meta["exposure_ms_readback"] == 200


def test_load_frame_with_meta_falls_back_to_frames_key(tmp_path: pathlib.Path) -> None:
    """探针帧是 ``frames`` (n,H,W) 且没有 params —— 与采集脚本帧共用同一读法。"""
    stack = np.zeros((2, 3, 4), dtype=np.uint8)
    stack[0] = 7
    path = tmp_path / "probe.npz"
    np.savez_compressed(path, frames=stack)

    got, meta = load_frame_with_meta(path)

    assert np.array_equal(got, np.full((3, 4), 7.0))
    assert meta == {}


def test_load_frame_with_meta_tolerates_bad_params(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, frame=np.zeros((2, 2)), params="not json {")

    got, meta = load_frame_with_meta(path)

    assert got.shape == (2, 2)
    assert meta == {}


# ---------------------------------------------------------------------------
# 边缘过渡宽度
# ---------------------------------------------------------------------------
def test_crossing_width_matches_synthetic_ramp() -> None:
    """平台 10 -> 底 0, 在 50..70 线性下降: 10-90 % 跨度应是 16 px。"""
    prof = np.empty(120)
    prof[:50] = 10.0
    prof[50:71] = np.linspace(10.0, 0.0, 21)
    prof[71:] = 0.0
    idx = np.arange(prof.size)

    width = _crossing_width(prof, idx, floor=0.0, contrast=10.0, sign=1.0)

    assert width == pytest.approx(16.0, abs=1.5)


def test_crossing_width_is_nan_without_contrast() -> None:
    prof = np.full(50, 3.0)
    assert np.isnan(_crossing_width(prof, np.arange(50), 3.0, 0.0, 1.0))


def _square_frame(side: int, size: int = 1200, sigma: float = 0.0,
                  level: float = 12.0) -> np.ndarray:
    """居中方斑 (level counts, 底 0); ``sigma>0`` 时用高斯模糊把边缘变软。"""
    c = size // 2
    lo, hi = c - side // 2, c + side // 2
    img = np.zeros((size, size), dtype=float)
    img[lo:hi, lo:hi] = level
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(img, sigma)
    return img


def test_edge_transition_separates_sharp_from_soft_edges() -> None:
    side = 400
    center = (600.0, 600.0)
    gaps = dict(inner_gap=60.0, outer_gap=140.0, span=120.0)

    sharp = edge_transition(_square_frame(side), center, side, **gaps)
    soft = edge_transition(_square_frame(side, sigma=8.0), center, side, **gaps)

    for name in ("left", "right"):
        assert sharp[name]["width_px"] < 5.0
        # 高斯 σ=8 的 10-90 % 宽度 = 2*1.2816*8 ≈ 20.5 px
        assert 12.0 < soft[name]["width_px"] < 30.0
        assert soft[name]["width_px"] > sharp[name]["width_px"] + 8.0
        assert soft[name]["width_20_80_px"] < soft[name]["width_px"]


def test_edge_transition_is_nan_when_there_is_no_square() -> None:
    flat = np.zeros((1200, 1200))
    out = edge_transition(flat, (600.0, 600.0), 400, inner_gap=60.0, outer_gap=140.0)

    assert np.isnan(out["left"]["width_px"])
    assert np.isnan(out["right"]["width_px"])


def test_edge_transition_platform_and_floor_are_read_off_the_frame() -> None:
    out = edge_transition(_square_frame(400, level=12.0), (600.0, 600.0), 400,
                          inner_gap=60.0, outer_gap=140.0)

    assert out["right"]["platform"] == pytest.approx(12.0, abs=0.5)
    assert out["right"]["floor"] == pytest.approx(0.0, abs=0.5)


# ---------------------------------------------------------------------------
# 未整形光斑
# ---------------------------------------------------------------------------
def test_psf_shape_recovers_gaussian_fwhm() -> None:
    """σ=4 px 的高斯, FWHM = 2*sqrt(2 ln 2)*σ ≈ 9.4 px。"""
    n, sigma, peak = 201, 4.0, 180.0
    y, x = np.mgrid[0:n, 0:n]
    img = peak * np.exp(-((x - 100.0) ** 2 + (y - 100.0) ** 2) / (2 * sigma ** 2))

    got = psf_shape(img, half=40)

    assert got["center"] == (100.0, 100.0)
    assert got["fwhm_x"] == pytest.approx(9.4, abs=1.0)
    assert got["fwhm_y"] == pytest.approx(9.4, abs=1.0)
    assert got["peak"] > 100.0


def test_fwhm_is_nan_when_nothing_crosses_half_max() -> None:
    assert np.isnan(_fwhm(np.zeros(10)))


# ---------------------------------------------------------------------------
# JSON 安全
# ---------------------------------------------------------------------------
def test_num_or_none_maps_non_finite_to_none() -> None:
    assert _num_or_none(float("nan")) is None
    assert _num_or_none(None) is None
    assert _num_or_none("abc") is None
    assert _num_or_none("2.5") == 2.5
