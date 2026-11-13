"""光路复用探针 —— 纯函数测试。

只测 :mod:`ao_shaping.tools.slm.transfer.optical_probe` 的纯函数部分:
``probe_from_latest`` / ``same_path`` / ``format_probe_verdict``, 平场斑的两条
形状判据 (归一化互相关 + 二阶矩 :func:`flat_shape`), 以及 workflow 的
``reuse_latest`` 计划开关。**不打开任何设备** (相机/SLM), 全部用合成夹具。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    latest = _write_latest(
        tmp_path / "latest", origin=(12.5, -3.25), sidecar=ref_sidecar
    )
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
        tmp_path / "latest",
        origin=(1.0, 1.0),
        sidecar={"exposure_range_ms": [0.1, 40.0]},
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
    defaults = dict(
        zero_xy=(10.0, 20.0),
        exposure_range_ms=(0.02, 25.0),
        camera_format="MONO8",
        bit_depth=8,
    )
    defaults.update(kw)
    return probe.OpticalProbe(**defaults)


def test_same_path_accepts_identical_path():
    assert probe.same_path(P(), P()) is True


@pytest.mark.parametrize(
    "delta,expected",
    [
        ((0.0, 0.0), True),
        ((5.0, 0.0), True),  # 正好在边界上 → 视作没变
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
        (25.0, True),  # 与参考一致
        (25.0 * 1.5, True),  # 正好 1.5 倍 → 边界上, 视作没变
        (25.0 / 1.5, True),  # 正好 1/1.5
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
    assert (
        probe.same_path(P(exposure_range_ms=(0.02, 50.0)), P(), tol_exposure_factor=3.0)
        is True
    )


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
        False,
        current=P(zero_xy=(40.0, 60.0), exposure_range_ms=(0.02, 99.0)),
        reference=P(),
    )
    assert "光路已变" in out
    assert "40.0" in out and "60.0" in out
    assert "99" in out and "25" in out
    assert len(out) < 120


def test_format_probe_skip_line_wraps_reason():
    """跳过原因 → 「光路复用：跳过（原因） → 走正常全流程」。"""
    out = probe.format_probe_skip_line(probe.REUSE_SKIP_DISABLED)
    assert out == "光路复用：跳过（光路复用未开启） → 走正常全流程"


def test_format_probe_skip_line_empty_reason_falls_back():
    """空/空白原因 → 无原因的兜底文案, 绝不抛。"""
    assert probe.format_probe_skip_line("") == "光路复用：跳过 → 走正常全流程"
    assert probe.format_probe_skip_line("   ") == "光路复用：跳过 → 走正常全流程"
    assert probe.format_probe_skip_line(None) == "光路复用：跳过 → 走正常全流程"


# ---------------------------------------------------------------------------
# 二阶矩形状判据 (flat_shape / compare_flat_shape / assess_reuse 的 shape 键)
#
# 这些用例是**判别力**测试, 不是冒烟测试: 每条都对着 2026-11-11 合成敏感性扫描
# 里 NCC 瞎掉的那一类扰动 (拉伸 / 旋转 / 离焦代理), 要求形状判据把它们拦下来,
# 同时要求纯平移与纯增益**不**被拦下来 (那两条正是 NCC 的强项)。
# 合成斑的参数照当天实测平场帧取: sigma_x=11.4 / sigma_y=11.7 px (aspect 1.028),
# 峰 73 counts, 背景 0.3, 噪声 0.6 counts (MONO8 @0.02 ms 量级)。
# ---------------------------------------------------------------------------

#: 合成斑的基准标准差 (px), 与实测平场斑同量级。
_SPOT_SIGMA = (11.4, 11.7)


def _spot(
    *,
    n: int = 420,
    centre: tuple[float, float] = (210.0, 210.0),
    sigma: tuple[float, float] = _SPOT_SIGMA,
    theta_deg: float = 0.0,
    amp: float = 73.0,
    background: float = 0.3,
    noise: float = 0.6,
    seed: int = 7,
):
    """高斯斑 + 常数背景 + 固定种子噪声 (纯 numpy, 不碰设备)。"""
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    dx = xx - centre[0]
    dy = yy - centre[1]
    t = float(np.radians(theta_deg))
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    g = np.exp(-0.5 * ((u / sigma[0]) ** 2 + (v / sigma[1]) ** 2))
    rng = np.random.default_rng(seed)
    return amp * g + background + rng.normal(0.0, noise, size=(n, n))


def _shape_of(cur, ref):
    """两帧走生产路径 :func:`compare_flat_frames`, 返回 (比对, 形状比对)。"""
    comparison = probe.compare_flat_frames(cur, ref)
    assert comparison.shape_comparison is not None
    return comparison, comparison.shape_comparison


def test_flat_shape_reads_a_circular_spot_as_round():
    m = probe.flat_shape(_spot(seed=11)[170:251, 170:251])
    assert m.sigma_g > 0.0
    # 近圆斑: 椭圆度接近 1, 且 sigma_x ~ sigma_y。
    assert m.aspect == pytest.approx(1.0, abs=0.08)
    assert m.sigma_x == pytest.approx(m.sigma_y, rel=0.08)
    assert -90.0 < m.theta_deg <= 90.0
    d = m.to_dict()
    assert set(d) == {"sigma_g", "sigma_x", "sigma_y", "aspect", "theta_deg"}


def test_flat_shape_is_scale_invariant_in_ratio_but_not_in_gain():
    ref = _spot(seed=3)[170:251, 170:251]
    cur = probe.flat_shape(ref * 3.0)
    base = probe.flat_shape(ref)
    # 亮度整体乘 3 不该改二阶矩 (强度加权后归一化掉了)。
    assert cur.sigma_g == pytest.approx(base.sigma_g, rel=1e-6)
    assert cur.aspect == pytest.approx(base.aspect, rel=1e-6)


def test_flat_shape_degenerate_returns_nan_and_never_raises():
    for bad in (np.zeros((64, 64)), -np.ones((64, 64)), np.full((64, 64), np.nan)):
        m = probe.flat_shape(bad)
        assert all(
            np.isnan(v)
            for v in (m.sigma_g, m.sigma_x, m.sigma_y, m.aspect, m.theta_deg)
        )
    # 非 2D / 空图也不抛。
    assert np.isnan(probe.flat_shape(np.zeros(10)).sigma_g)
    assert np.isnan(probe.flat_shape(np.zeros((0, 0))).sigma_g)


def test_compare_flat_shape_accepts_translation():
    ref = _spot()
    _, sc = _shape_of(_spot(centre=(240.0, 210.0)), ref)
    assert sc.known and sc.within_tolerance
    assert sc.size_ratio == pytest.approx(1.0, abs=0.02)
    assert sc.aspect_mismatch == pytest.approx(1.0, abs=0.02)


def test_compare_flat_shape_rejects_large_translation_on_drift_not_shape():
    # 60 px 平移: 形状仍然判「同一条光路」(平移由 assess_reuse 的漂移上限管,
    # 不是形状管) —— 这条是把两个判据的**分工**钉住, 防止将来互相重叠。
    ref = _spot()
    _, sc = _shape_of(_spot(centre=(270.0, 210.0)), ref)
    assert sc.within_tolerance is True


def test_compare_flat_shape_accepts_gain_only_change():
    ref = _spot()
    for factor, amp in ((0.3, 73.0 * 0.3), (3.0, 73.0 * 3.0)):
        _, sc = _shape_of(_spot(amp=amp, background=0.3 * factor), ref)
        assert sc.within_tolerance is True, factor
        assert sc.size_ratio == pytest.approx(1.0, abs=0.02), factor


def test_compare_flat_shape_rejects_anisotropic_stretch():
    # 20% 各向异性拉伸: NCC 0.982 (放行), 椭圆度失配 1.36 (拦下)。
    ref = _spot()
    comparison, sc = _shape_of(_spot(sigma=(11.4 * 1.20, 11.7 * 0.83)), ref)
    assert sc.within_tolerance is False
    assert sc.size_ratio == pytest.approx(1.03, abs=0.05)
    assert sc.aspect_mismatch > probe.DEFAULT_MAX_ASPECT_MISMATCH
    assert comparison.similarity > probe.DEFAULT_MIN_FLAT_SIMILARITY


def test_compare_flat_shape_rejects_rotation_of_elliptical_spot():
    # aspect 1.49 的椭圆斑转 25°: NCC 0.983 (放行), 夹角 25.1° > 15° (拦下)。
    ref = _spot(sigma=(15.0, 10.0))
    comparison, sc = _shape_of(_spot(sigma=(15.0, 10.0), theta_deg=25.0), ref)
    assert sc.angle_significant is True
    assert sc.axis_tilt_deg == pytest.approx(25.0, abs=1.0)
    assert sc.aspect_mismatch == pytest.approx(1.0, abs=0.02)
    assert sc.within_tolerance is False
    assert comparison.similarity > probe.DEFAULT_MIN_FLAT_SIMILARITY


def test_compare_flat_shape_keeps_12deg_rotation_under_the_cap():
    # 扫描标定: 12° 是「能看见但先不否决」的那一档 (上限 15°), 钉住边界。
    ref = _spot(sigma=(15.0, 10.0))
    _, sc = _shape_of(_spot(sigma=(15.0, 10.0), theta_deg=12.0), ref)
    assert sc.angle_significant is True
    assert sc.axis_tilt_deg == pytest.approx(12.0, abs=1.0)
    assert sc.within_tolerance is True


def test_compare_flat_shape_ignores_angle_for_near_circular_spot():
    """近圆斑转 25°: 角度**不参与判比**, 于是不会被误判成光路变了。

    实测平场斑 aspect 1.028, 这种斑的主轴方向完全由噪声决定 (转不转都一样),
    拿它否决等于用噪声判案。结论: 这类扰动**形状判据抓不到** —— 而且物理上
    「圆斑绕自己中心转」本来就不是光路变化, 该抓的是尺寸与椭圆度两条轴。
    """
    ref = _spot()
    _, sc = _shape_of(_spot(theta_deg=25.0), ref)
    assert sc.angle_significant is False
    assert sc.axis_tilt_deg > probe.DEFAULT_MAX_AXIS_TILT_DEG  # 数本身是噪声
    assert sc.within_tolerance is True


def test_compare_flat_shape_rejects_isotropic_widening_defocus_proxy():
    # 30% 各向同性放宽 (离焦代理): NCC 0.968 (放行), 尺寸比 1.22 > 1.12 (拦下)。
    ref = _spot()
    comparison, sc = _shape_of(_spot(sigma=(11.4 * 1.3, 11.7 * 1.3)), ref)
    assert sc.size_ratio > probe.DEFAULT_MAX_SIZE_RATIO
    assert sc.aspect_mismatch == pytest.approx(1.0, abs=0.02)
    assert sc.within_tolerance is False
    assert comparison.similarity > probe.DEFAULT_MIN_FLAT_SIMILARITY


def test_compare_flat_shape_rejects_isotropic_narrowing_too():
    # 斑**变小**同样是光路变了 (口径/焦距), 所以尺寸比是双向卡 (1/1.12 ~ 0.893)。
    ref = _spot()
    _, sc = _shape_of(_spot(sigma=(11.4 * 0.77, 11.7 * 0.77)), ref)
    assert sc.size_ratio < 1.0 / probe.DEFAULT_MAX_SIZE_RATIO
    assert sc.within_tolerance is False


def test_compare_flat_shape_is_symmetric_in_aspect_mismatch():
    a = probe.flat_shape(_spot(sigma=(15.0, 10.0), seed=5)[170:251, 170:251])
    b = probe.flat_shape(_spot(seed=5)[170:251, 170:251])
    forward = probe.compare_flat_shape(a, b)
    backward = probe.compare_flat_shape(b, a)
    assert forward.aspect_mismatch == pytest.approx(backward.aspect_mismatch, rel=1e-9)
    assert forward.aspect_mismatch >= 1.0
    assert forward.size_ratio == pytest.approx(1.0 / backward.size_ratio, rel=1e-9)


def test_compare_flat_shape_fails_closed_on_degenerate_reference():
    ref = probe.flat_shape(np.zeros((64, 64)))
    cur = probe.flat_shape(_spot(seed=9)[170:251, 170:251])
    sc = probe.compare_flat_shape(cur, ref)
    assert sc.known is False
    assert sc.within_tolerance is False
    assert np.isnan(sc.size_ratio)


def test_compare_flat_frames_carries_shape_fields_and_old_ones():
    ref = _spot()
    comparison = probe.compare_flat_frames(_spot(centre=(230.0, 205.0)), ref)
    # 老字段一个都不能少、含义不变 (GUI 与 optical_probe.json 在读)。
    assert comparison.window_px == 2 * probe.DEFAULT_FLAT_HALF_WINDOW + 1
    assert -1.0 <= comparison.similarity <= 1.0
    assert len(comparison.drift_px) == 2
    assert comparison.shape is not None and comparison.reference_shape is not None
    d = comparison.to_dict()
    assert {"similarity", "drift_px", "window_px", "current_xy", "reference_xy"} <= set(
        d
    )
    assert {"shape", "reference_shape", "shape_comparison"} <= set(d)


def test_assess_reuse_needs_ncc_zero_and_shape_all_together():
    ref = _spot()
    p = P()
    # 形状变了 (各向异性拉伸) 但 NCC/漂移都过 → 必须判「不是同一条光路」。
    decision = probe.assess_reuse(
        p, p, current_frame=_spot(sigma=(11.4 * 1.2, 11.7 * 0.83)), reference_frame=ref
    )
    assert decision.same is False
    assert decision.criteria["flat"].startswith("平场斑相似度")
    assert decision.criteria["shape"].startswith("形状判据未过: 椭圆度失配")

    # 什么都没变 → 过, 且 criteria 把三条轴的数写出来给 GUI 看。
    decision = probe.assess_reuse(p, p, current_frame=ref, reference_frame=ref)
    assert decision.same is True
    assert "尺寸比" in decision.criteria["shape"]
    assert "(通过)" in decision.criteria["shape"]


def test_assess_reuse_criteria_keys_survive():
    ref = _spot()
    decision = probe.assess_reuse(
        P(),
        P(),
        current_frame=_spot(sigma=(11.4 * 1.3, 11.7 * 1.3)),
        reference_frame=ref,
    )
    keys = set(decision.criteria)
    assert {"flat", "shape", "exposure", "camera_format", "bit_depth"} <= keys
    assert decision.criteria["shape"].startswith("形状判据未过: 尺寸比")


def test_assess_reuse_without_frames_keeps_the_zero_fallback():
    # 没有平场帧 = 没有形状证据: 形状判据不参与, 老语义 (各轴 <= tol_zero_px) 不变。
    decision = probe.assess_reuse(P(zero_xy=(11.0, 21.0)), P())
    assert decision.same is True
    assert decision.flat is None
    assert "不可用" in decision.criteria["shape"]
    assert probe.assess_reuse(P(zero_xy=(100.0, 20.0)), P()).same is False


def test_assess_reuse_degenerate_frame_fails_closed_without_raising():
    # 全零帧 (没对准/快门没开) 不该抛异常, 也不该被读成「形状没变」。
    ref = _spot()
    decision = probe.assess_reuse(
        P(), P(), current_frame=np.zeros_like(ref), reference_frame=ref
    )
    assert decision.same is False
    assert "退化" in decision.criteria["shape"]
    # 两帧都退化时也一样 (NCC 那关先就不过, 形状这关给出「无证据」的说法)。
    both = probe.assess_reuse(
        P(), P(), current_frame=np.zeros_like(ref), reference_frame=np.zeros_like(ref)
    )
    assert both.same is False and "退化" in both.criteria["shape"]


def test_shape_dict_is_strict_json_safe():
    # 退化帧的 NaN 不许漏进 optical_probe.json (严格 JSON 解析器会直接拒绝)。
    ref = _spot()
    decision = probe.assess_reuse(
        P(), P(), current_frame=np.zeros_like(ref), reference_frame=ref
    )
    payload = json.dumps(decision.to_dict(), allow_nan=False)
    assert "NaN" not in payload
    restored = json.loads(payload)
    assert restored["criteria"]["shape"]
    assert restored["flat"]["shape"]["sigma_g"] is None  # NaN -> None
    # 正常帧的数照常落盘, 不被抹掉。
    good = json.dumps(probe.compare_flat_frames(ref, ref).to_dict(), allow_nan=False)
    assert json.loads(good)["shape"]["sigma_g"] > 0.0


def test_format_probe_verdict_adds_shape_only_when_shape_exists():
    ref = _spot()
    comparison = probe.compare_flat_frames(_spot(sigma=(11.4 * 1.3, 11.7 * 1.3)), ref)
    out = probe.format_probe_verdict(False, current=P(), reference=P(), flat=comparison)
    assert "尺寸比" in out and "椭圆度失配" in out and "主轴夹角" in out
    # 没有平场帧的老调用点: 文案必须逐字节不变。
    legacy = probe.format_probe_verdict(False, current=P(), reference=P())
    assert legacy.startswith("光路复用: 光路已变 (zero=")
    # 手工构造的旧式 FlatComparison (没有二阶矩) 也不许长出形状片段。
    hand = probe.FlatComparison(0.7, (1.0, -2.0), 129, (300.0, 200.0), (299.0, 202.0))
    out_hand = probe.format_probe_verdict(False, current=P(), reference=P(), flat=hand)
    assert "形状" not in out_hand and "尺寸比" not in out_hand


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


# ---------------------------------------------------------------------------
# workflow._write_reference_probe: 平场参照帧随侧车进 fit_out (再随 latest)
# ---------------------------------------------------------------------------


def _reference_probe_dirs(tmp_path, *, flat: bool):
    """一套 fit_out / dataset / checkpoint; ``flat`` 决定采集目录里有没有平场帧。"""
    dataset = tmp_path / "dataset"
    fit_out = tmp_path / "fit"
    dataset.mkdir()
    fit_out.mkdir()
    checkpoint = fit_out / "frozen_parameters.json"
    checkpoint.write_text(json.dumps({"origin": [10.0, 20.0]}), encoding="utf-8")
    (dataset / probe.FLAT_FROZEN_SIDECAR).write_text(
        json.dumps(
            {
                "exposure_range_ms": [0.02, 25.0],
                "camera_format": "MONO8",
                "bit_depth": 8,
                "slm_serial": "SLM-A",
            }
        ),
        encoding="utf-8",
    )
    if flat:
        assert (
            probe.save_flat_frame(dataset, np.zeros((4, 4), np.float32))
            == probe.FLAT_FRAME_FILE
        )
    return dataset, fit_out, checkpoint


def test_write_reference_probe_copies_flat_frame_into_fit_out(tmp_path):
    """采集落盘过平场帧 → 复制进 fit_out 且侧车登记 ``flat_frame`` 键。"""
    from ao_shaping.tools.slm.transfer.workflow import _write_reference_probe

    dataset, fit_out, checkpoint = _reference_probe_dirs(tmp_path, flat=True)
    _write_reference_probe(fit_out, dataset, checkpoint)

    assert (fit_out / probe.FLAT_FRAME_FILE).is_file()
    record = json.loads(
        (fit_out / probe.FLAT_FROZEN_SIDECAR).read_text(encoding="utf-8")
    )
    assert record["flat_frame"] == probe.FLAT_FRAME_FILE
    # 既有键语义不变 (加法改动)。
    assert record["origin"] == [10.0, 20.0]
    assert record["exposure_range_ms"] == [0.02, 25.0]


def test_write_reference_probe_omits_flat_frame_when_missing(tmp_path):
    """采集没落盘平场帧 → fit_out 里没有帧文件, 侧车也不写 ``flat_frame`` 键。"""
    from ao_shaping.tools.slm.transfer.workflow import _write_reference_probe

    dataset, fit_out, checkpoint = _reference_probe_dirs(tmp_path, flat=False)
    _write_reference_probe(fit_out, dataset, checkpoint)

    assert not (fit_out / probe.FLAT_FRAME_FILE).exists()
    record = json.loads(
        (fit_out / probe.FLAT_FROZEN_SIDECAR).read_text(encoding="utf-8")
    )
    assert "flat_frame" not in record


# ---------------------------------------------------------------------------
# zero_drift_verdict (#243 前置二): 5 < 20 黄警 < 40 硬上限 三档
# ---------------------------------------------------------------------------


def test_zero_drift_verdict_three_levels():
    assert probe.zero_drift_verdict((10.0, 10.0), (10.0, 10.0))[0] == "ok"
    assert probe.zero_drift_verdict((35.0, 10.0), (10.0, 10.0))[0] == "warn"
    assert probe.zero_drift_verdict((60.0, 10.0), (10.0, 10.0))[0] == "hard"


def test_zero_drift_verdict_uses_axis_max_like_assess_reuse():
    level, _ = probe.zero_drift_verdict((10.0, 35.0), (10.0, 10.0))
    assert level == "warn"
    level, _ = probe.zero_drift_verdict((10.0, 55.0), (10.0, 10.0))
    assert level == "hard"


def test_zero_drift_verdict_boundaries_are_exclusive():
    assert probe.zero_drift_verdict((30.0, 10.0), (10.0, 10.0))[0] == "ok"
    assert probe.zero_drift_verdict((50.0, 10.0), (10.0, 10.0))[0] == "warn"


def test_zero_drift_verdict_without_reference_never_alarms():
    assert probe.zero_drift_verdict((500.0, 500.0), None)[0] == "ok"
    assert probe.zero_drift_verdict((500.0, 500.0), "not-a-pair")[0] == "ok"
