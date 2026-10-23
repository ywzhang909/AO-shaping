"""幂律相位响应的离线验证（数值取自 2026-10-23 在焦实测报告）。"""

from __future__ import annotations

import numpy as np
import pytest

from ao_shaping.utils.slm.phase_response import (
    MEASURED_HW_20261023,
    PhaseResponseError,
    PowerLawPhaseResponse,
    describe_linear_error,
    load_phase_response,
    save_phase_response,
)

TWO_PI = 2.0 * np.pi


def test_measured_response_matches_report_numbers():
    response = MEASURED_HW_20261023
    assert response.depth_at_full == pytest.approx(1.845 * np.pi, rel=1e-12)
    assert response.depth_ratio == pytest.approx(0.9225, rel=1e-4)
    # 2π 需要 1130 灰度左右，超出面板 1023 ⇒ 不可达
    assert response.gray_for_two_pi() == pytest.approx(1130, abs=15)
    assert response.gray_for_two_pi() > response.gray_max


def test_phase_of_gray_endpoints():
    response = MEASURED_HW_20261023
    assert response.phase_of_gray(0) == pytest.approx(0.0, abs=1e-12)
    assert response.phase_of_gray(1023) == pytest.approx(1.845 * np.pi, rel=1e-12)
    # 超出灰度上限的值按上限处理
    assert response.phase_of_gray(2000) == pytest.approx(response.phase_of_gray(1023))


def test_round_trip_within_reachable_range():
    response = MEASURED_HW_20261023
    phases = np.linspace(0.0, 1.8 * np.pi, 50)
    gray, clamped = response.gray_of_phase(phases)
    assert not clamped.any()
    assert np.allclose(response.phase_of_gray(gray), phases, atol=1e-9)


def test_unreachable_phase_is_clamped_and_flagged():
    response = MEASURED_HW_20261023
    gray, clamped = response.gray_of_phase([1.0, 1.9 * np.pi, TWO_PI])
    assert clamped.tolist() == [False, True, True]
    assert gray[1] == pytest.approx(1023.0, rel=1e-9)
    assert gray[2] == pytest.approx(1023.0, rel=1e-9)


def test_linear_mapping_error_is_large_at_small_gray():
    """线性 2π↔gray_max 换算在小灰度段偏高（γ<1），满量程段偏低（深度不足 2π）。"""
    response = MEASURED_HW_20261023
    small = describe_linear_error(response, gray=158)      # 约 1 rad 的请求
    large = describe_linear_error(response, gray=1023)
    assert small["relative_error"] > 0.15                 # 低灰度段明显偏高
    assert large["relative_error"] < 0                      # 满量程偏低（1.845π < 2π）


def test_local_gain_decreases_with_gray():
    response = MEASURED_HW_20261023
    gains = [response.local_gain(g) for g in (50, 200, 800)]
    assert gains[0] > gains[1] > gains[2]


def test_json_round_trip_and_validation(tmp_path):
    path = save_phase_response(MEASURED_HW_20261023, tmp_path / "resp.json")
    loaded = load_phase_response(path)
    assert loaded.depth_at_full == pytest.approx(MEASURED_HW_20261023.depth_at_full)
    assert loaded.gamma == pytest.approx(MEASURED_HW_20261023.gamma)

    with pytest.raises(PhaseResponseError):
        PowerLawPhaseResponse(depth_at_full=0.0, gamma=0.8)
    with pytest.raises(PhaseResponseError):
        PowerLawPhaseResponse(depth_at_full=1.0, gamma=0.0)
