"""「光路受阻」的判定 —— :func:`ao_shaping.utils.image.frame_analysis.assess_spot`.

为什么这些测试存在
------------------
光路被挡住时, 画面里**没有任何局域结构**, 而「找光斑」的流程看到的正是同一张图:
焦面探测报「找不到焦面」, 离焦探针把纯噪声记成 30 行测量结果。两种表现都**看不出**
光路已经断了, 于是人只能去翻帧才发现。

所以这里锁住的是**三种故障的区分**:

1. 干净的光斑 ⇒ 可用;
2. 平场读出噪声 ⇒ **光路受阻** (必须带上三段光路的检查清单);
3. 整幅渐变 / 弥散光斑 ⇒ 光**到了**但没法评分 (不是受阻, 是没聚焦)。

第 3 条最容易写错: 「亮」不等于「有光斑」。整幅渐变的峰值是满量程, 却没有任何
可评分的东西 —— 把它判成「光路受阻」会让操作者去查三段光路, 而真正的问题是焦面。
"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.image.frame_analysis import (
    BLOCKED_PATH_CHECKLIST,
    BLOCKED_PATH_REASON,
    MIN_BRIGHT_FILL,
    assess_spot,
)

#: 2026-10-20 正常实验系统的实测值: 250x250 ROI 上峰值 42 counts、FWHM 12.9 px。
HEALTHY_PEAK = 42.0
HEALTHY_FWHM_PX = 12.9

#: 2026-10-21 光路受阻时的实测签名: 峰值停在 1-2 counts, FWHM ≈ 整个画幅
#: (2592x1944 上实测 2267 px)。
BLOCKED_PEAK_COUNTS = (1.0, 2.0)


def gaussian_frame(
    *,
    shape: tuple[int, int] = (250, 250),
    peak: float = HEALTHY_PEAK,
    fwhm_px: float = HEALTHY_FWHM_PX,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """一个高斯光斑 (FWHM 精确可控)。"""
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    cy = height / 2.0 if cy is None else cy
    cx = width / 2.0 if cx is None else cx
    sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return peak * np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2)
    )


def read_noise_frame(
    *, shape: tuple[int, int] = (250, 250), sigma: float = 0.29, seed: int = 0
) -> np.ndarray:
    """平场读出噪声 (零均值高斯, σ 落在本实验系统这台相机的实测值上)。"""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, shape)


def prepared_frame(frame: np.ndarray) -> np.ndarray:
    """``prepare_frame`` 的口径 (减中位数 + 裁负) —— 离焦探针存的就是这种帧。"""
    return np.clip(frame - np.median(frame), 0.0, None)


class TestHealthySpot:
    """正常光斑必须**永远**判成可用 —— 判错就是白跑一轮实验系统机时。"""

    def test_a_clean_spot_is_usable(self):
        got = assess_spot(gaussian_frame())
        assert got.usable is True
        assert got.blocked is False
        assert got.reason == "光斑可用"

    def test_the_measured_peak_and_width_are_reported(self):
        got = assess_spot(gaussian_frame())
        assert got.peak == pytest.approx(HEALTHY_PEAK, rel=0.02)
        assert got.fwhm_px == pytest.approx(HEALTHY_FWHM_PX, abs=2.5)
        assert got.frame_px == 250

    def test_a_detector_pedestal_does_not_hide_the_spot(self):
        """相机本底 (offset) 会整体抬高峰值, 但结构还在, 所以仍判可用。"""
        got = assess_spot(gaussian_frame() + 100.0)
        assert got.usable is True
        assert got.blocked is False
        # 判「够不够亮」用的是**背景之上**的余量, 不是原始峰值。
        assert got.excess == pytest.approx(HEALTHY_PEAK, rel=0.05)
        assert got.peak > HEALTHY_PEAK

    def test_a_dim_but_structured_spot_is_not_a_blockage(self):
        """光很弱但结构完整 ⇒ 曝光不足, 不是光路断了。"""
        got = assess_spot(gaussian_frame(peak=3.0))
        assert got.blocked is False
        assert got.usable is True
        assert "偏暗" in got.reason


class TestBlockedPath:
    """平场读出噪声 = 光路受阻。"""

    def test_a_flat_noise_frame_is_blocked(self):
        got = assess_spot(read_noise_frame())
        assert got.blocked is True
        assert got.usable is False

    def test_the_reason_names_the_three_path_segments(self):
        got = assess_spot(read_noise_frame())
        assert got.reason == BLOCKED_PATH_REASON
        assert "光束没有到达相机" in got.reason
        for segment in ("SLM", "2f", "CCD"):
            assert segment in got.reason
        assert BLOCKED_PATH_CHECKLIST in got.reason

    def test_the_peak_sits_on_the_read_noise_floor(self):
        got = assess_spot(read_noise_frame())
        assert BLOCKED_PEAK_COUNTS[0] <= got.excess <= BLOCKED_PEAK_COUNTS[1] + 3.0

    def test_the_full_size_frame_spans_the_whole_frame(self):
        """2026-10-21 实测: 2592x1944 上 FWHM ≈ 2267 px (整个画幅)。"""
        got = assess_spot(read_noise_frame(shape=(1944, 2592), seed=3))
        assert got.blocked is True
        assert got.fwhm_px > 0.8 * got.frame_px

    def test_a_prepared_noise_frame_is_blocked_too(self):
        """离焦探针存的是减过中位数的帧 —— 那种帧同样必须判成受阻。"""
        got = assess_spot(prepared_frame(read_noise_frame(seed=7)))
        assert got.blocked is True
        assert got.usable is False

    def test_a_black_frame_is_blocked(self):
        got = assess_spot(np.zeros((250, 250)))
        assert got.blocked is True
        assert got.usable is False

    def test_a_pedestal_plus_noise_is_blocked(self):
        """带本底的暗帧 (相机在, 光没到) 同样是受阻。"""
        got = assess_spot(100.0 + read_noise_frame(seed=11))
        assert got.blocked is True
        assert got.excess < 5.0


class TestLightButNotFocused:
    """光**到了**但没法评分 —— 这是另一种故障, 不能说成「光路断了」。"""

    def test_a_full_frame_ramp_is_not_a_blockage(self):
        """整幅渐变: 峰值是满量程, 却没有任何局域光斑。"""
        ramp = np.tile(np.linspace(0.0, 255.0, 250), (250, 1))
        got = assess_spot(ramp)
        assert got.blocked is False
        assert got.usable is False
        assert "没有可评分" in got.reason

    def test_a_grossly_defocused_spot_is_not_a_blockage(self):
        """光铺满整幅 (严重离焦) ⇒ 光到了, 只是没聚焦。"""
        got = assess_spot(gaussian_frame(fwhm_px=236.0, peak=20.0))
        assert got.blocked is False
        assert got.usable is False

    def test_a_scattered_pattern_is_not_a_blockage_when_it_is_bright(self):
        """亮但散 (例如镜头没收到光而杂散光很强) —— 不是「没光」。"""
        rng = np.random.default_rng(5)
        frame = rng.uniform(0.0, 255.0, (250, 250))
        got = assess_spot(frame)
        assert got.blocked is False


class TestStructureMetrics:
    """判定用的中间量必须能被人复核。"""

    def test_the_bright_region_of_a_spot_is_one_block(self):
        got = assess_spot(gaussian_frame())
        assert got.fill >= MIN_BRIGHT_FILL
        assert got.span_fraction < 0.2

    def test_noise_bright_pixels_are_scattered(self):
        got = assess_spot(read_noise_frame(seed=13))
        assert got.fill < MIN_BRIGHT_FILL

    def test_the_dict_form_is_json_ready(self):
        import json

        payload = assess_spot(gaussian_frame()).as_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload
        assert set(payload) >= {
            "usable", "blocked", "peak", "excess", "fwhm_px", "frame_px", "reason",
        }

    def test_a_non_2d_frame_is_rejected(self):
        with pytest.raises(ValueError):
            assess_spot(np.zeros((4, 4, 3)))
