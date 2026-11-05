"""光路复用探针 —— 纯函数测试。

只测 :mod:`ao_shaping.tools.slm.transfer.optical_probe` 的纯函数部分:
``probe_from_latest`` / ``same_path`` / ``format_probe_verdict``, 以及 workflow
的 ``reuse_latest`` 计划开关。**不打开任何设备** (相机/SLM), 全部用合成夹具。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ao_shaping.tools.slm.transfer import optical_probe as probe
from ao_shaping.tools.slm.transfer.workflow import TransferSession

#: 与 workflow 计划里的 stage 名一致。
OPTICAL_PROBE_STAGE = "optical_probe"


# ---------------------------------------------------------------------------
# 夹具: 合成 latest 目录
# ---------------------------------------------------------------------------


def _write_latest(latest_dir: Path, *, origin=(32.0, 48.0), sidecar=None):
    """写一个含 frozen_parameters.json (带 origin) + 可选探针侧车的 latest 目录。

    ``origin`` 原样写入 —— 允许传 None / 标量 / 元组 等「坏」值来模拟损坏的
    frozen_parameters.json。
    """
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "frozen_parameters.json").write_text(
        json.dumps({"origin": origin}), encoding="utf-8"
    )
    if sidecar is not None:
        (latest_dir / probe.FLAT_FROZEN_SIDECAR).write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    return latest_dir


@pytest.fixture
def ref_sidecar():
    return {
        "exposure_range_ms": [0.02, 25.0],
        "camera_format": "MONO8",
        "bit_depth": 8,
        "slm_serial": "SLM-A",
    }


# ---------------------------------------------------------------------------
# probe_from_latest: 从合成 latest 目录解析参照
# ---------------------------------------------------------------------------


def test_probe_from_latest_parses_origin_and_sidecar(tmp_path, ref_sidecar):
    latest = _write_latest(tmp_path / "latest", origin=(12.5, -3.25), sidecar=ref_sidecar)
    p = probe.probe_from_latest(latest)
    assert p is not None
    assert p.zero_xy == pytest.approx((12.5, -3.25))
    assert p.exposure_range_ms == pytest.approx((0.02, 25.0))
    assert p.camera_format == "MONO8"
    assert p.bit_depth == 8


def test_probe_from_latest_returns_none_when_latest_missing(tmp_path):
    assert probe.probe_from_latest(tmp_path / "nope") is None


def test_probe_from_latest_returns_none_when_frozen_missing(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert probe.probe_from_latest(d) is None


def test_probe_from_latest_returns_none_on_bad_origin(tmp_path):
    for i, bad in enumerate([None, [1, 2, 3], ["x", None], "12,13"]):
        latest = _write_latest(tmp_path / f"bad-{i}", origin=bad)
        assert probe.probe_from_latest(latest) is None, bad


def test_probe_from_latest_degrades_to_frozen_when_sidecar_missing(tmp_path):
    # 无侧车 → 格式/位深为 None (在 same_path 里即「不可比」), 但 origin 与
    # 曝光区间仍能读 (曝光无来源时回落到默认传感器区间)。
    latest = _write_latest(tmp_path / "latest", origin=(5.0, 6.0))
    p = probe.probe_from_latest(latest)
    assert p is not None
    assert p.zero_xy == pytest.approx((5.0, 6.0))
    assert p.camera_format is None
    assert p.bit_depth is None
    assert p.exposure_range_ms == pytest.approx((0.011, 10_000.0))


def test_probe_from_latest_uses_sidecar_exposure_over_provenance(tmp_path):
    latest = _write_latest(
        tmp_path / "latest", origin=(1.0, 1.0), sidecar={"exposure_range_ms": [0.1, 40.0]}
    )
    (latest / "provenance.json").write_text(
        json.dumps({"exposure_range_ms": [0.2, 20.0]}), encoding="utf-8"
    )
    p = probe.probe_from_latest(latest)
    assert p is not None
    assert p.exposure_range_ms == pytest.approx((0.1, 40.0))
    assert p.camera_format is None  # 侧车没给格式字段


def test_probe_from_latest_reads_exposure_from_provenance_when_sidecar_bare(
    tmp_path,
):
    latest = _write_latest(
        tmp_path / "latest", origin=(1.0, 1.0), sidecar={"camera_format": "MONO8"}
    )
    (latest / "provenance.json").write_text(
        json.dumps({"exposure_range_ms": [0.5, 60.0]}), encoding="utf-8"
    )
    p = probe.probe_from_latest(latest)
    assert p is not None
    assert p.exposure_range_ms == pytest.approx((0.5, 60.0))
    assert p.camera_format == "MONO8"


def test_probe_from_latest_falls_back_to_default_exposure(tmp_path):
    # 完全没有曝光来源 → 回落到默认传感器区间, 不该崩溃。
    latest = _write_latest(tmp_path / "latest", origin=(1.0, 1.0), sidecar={})
    p = probe.probe_from_latest(latest)
    assert p is not None
    assert p.exposure_range_ms == pytest.approx((0.011, 10_000.0))


# ---------------------------------------------------------------------------
# same_path: 边界判定 (纯函数, 不读文件, 绝不抛)
# ---------------------------------------------------------------------------


def P(**kw):
    defaults = dict(zero_xy=(10.0, 20.0), exposure_range_ms=(0.02, 25.0),
                    camera_format="MONO8", bit_depth=8)
    defaults.update(kw)
    return probe.OpticalProbe(**defaults)


def test_same_path_accepts_identical_path():
    assert probe.same_path(P(), P()) is True


@pytest.mark.parametrize(
    "delta,expected",
    [
        ((0.0, 0.0), True),
        ((5.0, 0.0), True),   # 正好在边界上 → 视作没变
        ((0.0, 5.0), True),
        ((5.0, 5.0), True),
        ((5.1, 0.0), False),  # 越过容差
        ((0.0, -5.1), False),
    ],
)
def test_same_path_zero_drift_boundary(delta, expected):
    cur = P(zero_xy=(10.0 + delta[0], 20.0 + delta[1]))
    assert probe.same_path(cur, P()) is expected


@pytest.mark.parametrize(
    "cur_hi,expected",
    [
        (25.0, True),          # 与参考一致
        (25.0 * 1.5, True),    # 正好 1.5 倍 → 边界上, 视作没变
        (25.0 / 1.5, True),    # 正好 1/1.5
        (25.0 * 1.51, False),  # 超过 1.5 倍
        (25.0 / 1.51, False),
    ],
)
def test_same_path_exposure_factor_boundary(cur_hi, expected):
    cur = P(exposure_range_ms=(0.02, cur_hi))
    assert probe.same_path(cur, P()) is expected


def test_same_path_requires_exact_format():
    assert probe.same_path(P(camera_format="MONO8"), P(camera_format="MONO8")) is True
    assert probe.same_path(P(camera_format="MONO12"), P(camera_format="MONO8")) is False
    assert probe.same_path(P(camera_format=None), P(camera_format="MONO8")) is False
    assert probe.same_path(P(camera_format="MONO8"), P(camera_format=None)) is False


def test_same_path_requires_exact_bit_depth():
    assert probe.same_path(P(bit_depth=8), P(bit_depth=8)) is True
    assert probe.same_path(P(bit_depth=12), P(bit_depth=8)) is False
    assert probe.same_path(P(bit_depth=None), P(bit_depth=8)) is False


def test_same_path_none_reference_is_false():
    assert probe.same_path(P(), None) is False


def test_same_path_missing_fields_handling():
    # 当前探针缺曝光 → 判不可比 (False)。
    assert probe.same_path(P(exposure_range_ms=None), P()) is False
    # 参照缺曝光时 same_path 跳过曝光判据 (ref_hi=0), 不因此判 False ——
    # 格式/位深仍须完整一致才会 True。
    assert probe.same_path(P(), P(exposure_range_ms=None)) is True
    assert probe.same_path(P(camera_format=None), P(exposure_range_ms=None)) is False


def test_same_path_never_raises_on_garbage_zero():
    garbage = P(zero_xy=("x", None))
    assert probe.same_path(garbage, P()) is False
    assert probe.same_path(P(), garbage) is False


def test_same_path_custom_tolerances():
    assert probe.same_path(P(zero_xy=(20.0, 20.0)), P(), tol_zero_px=10.0) is True
    assert probe.same_path(P(zero_xy=(20.0, 20.0)), P(), tol_zero_px=5.0) is False
    assert probe.same_path(P(exposure_range_ms=(0.02, 50.0)), P(),
                           tol_exposure_factor=3.0) is True


# ---------------------------------------------------------------------------
# format_probe_verdict: 人类可读判决 (短文案)
# ---------------------------------------------------------------------------


def test_format_probe_verdict_reuse_string():
    out = probe.format_probe_verdict(True)
    assert "跳过采集+拟合" in out
    assert len(out) < 80


def test_format_probe_verdict_no_reference():
    out = probe.format_probe_verdict(False, current=P(), reference=None)
    assert "无 latest 参照" in out
    assert "正常全流程" in out


def test_format_probe_verdict_missing_current():
    out = probe.format_probe_verdict(False, current=None, reference=P())
    assert "探针缺失" in out


def test_format_probe_verdict_changed_path_mentions_zero_and_exposure():
    out = probe.format_probe_verdict(
        False, current=P(zero_xy=(40.0, 60.0), exposure_range_ms=(0.02, 99.0)),
        reference=P(),
    )
    assert "光路已变" in out
    assert "40.0" in out and "60.0" in out
    assert "99" in out and "25" in out
    assert len(out) < 120


# ---------------------------------------------------------------------------
# workflow: reuse_latest 计划开关
# ---------------------------------------------------------------------------


def test_plan_includes_optical_probe_when_reuse_latest(tmp_path):
    config = TransferSession(tmp_path / "reuse", reuse_latest=True)
    assert OPTICAL_PROBE_STAGE in config.plan()["stages"]
    assert config.plan()["stages"].index(OPTICAL_PROBE_STAGE) == 0


def test_plan_omits_optical_probe_by_default(tmp_path):
    config = TransferSession(tmp_path / "default")
    assert OPTICAL_PROBE_STAGE not in config.plan()["stages"]


def test_reuse_latest_defaults_off(tmp_path):
    config = TransferSession(tmp_path / "default")
    assert config.reuse_latest is False