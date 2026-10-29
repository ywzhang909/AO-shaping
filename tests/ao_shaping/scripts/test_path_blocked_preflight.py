"""扫描前的**光路自检** (pre-flight) —— 受阻就别把噪声记成测量结果。

背景 (2026-10-21 实测)
-----------------------
光路被挡住时 (光束没进 2f 透镜 / 透镜没对准 / CCD 窗口被挡), 画面里**没有任何
局域结构**, 只剩读出噪声。而噪声帧照样能算出零级位置、光斑腰径、CV/PIB ——
于是:

* 焦面探测报的是「找不到焦面」, 与「焦面不在扫描范围里」完全一样;
* 离焦探针把 30 行**纯噪声**当成测量结果记下来, 看上去毫无异常。

人肉翻帧才发现光束根本没进透镜。所以两个扫描脚本都必须在**开跑之前**判一次:
受阻就中止, 并告诉操作者去查 光束→SLM 面板 → 2f 透镜 → CCD 三段光路。

这些测试不需要任何硬件: 判据是纯函数, 帧是合成的。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ao_shaping.utils.image.frame_analysis import (
    BLOCKED_PATH_REASON,
    assess_spot,
)
from scripts.slm.hw_gs_defocus_probe import (
    EXIT_PATH_BLOCKED,
    PathBlockedError,
    build_parser,
    build_plan,
    build_optics,
    preflight_assessment,
    raise_if_path_blocked,
)
from scripts.slm.run_gs_refine_zernike_sweep import PathBlockedError as SweepBlockedError

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "scripts" / "slm" / "hw_gs_defocus_probe.py"


def noise_frame(shape: tuple[int, int] = (250, 250), seed: int = 0) -> np.ndarray:
    """平场读出噪声 —— 光路受阻时画面里的东西。"""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.29, shape)


def spot_frame(shape: tuple[int, int] = (250, 250), peak: float = 42.0) -> np.ndarray:
    """一个正常的高斯光斑。"""
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    sigma = 12.9 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return peak * np.exp(
        -((yy - height / 2.0) ** 2 + (xx - width / 2.0) ** 2) / (2.0 * sigma**2)
    )


class TestSharedVerdict:
    """两个脚本用的是**同一个**判据 —— 免得两处说法不一致。"""

    def test_noise_is_blocked_and_a_spot_is_not(self):
        assert preflight_assessment(noise_frame()).blocked is True
        assert preflight_assessment(spot_frame()).blocked is False

    def test_the_reason_carries_the_checklist(self):
        assert "SLM" in BLOCKED_PATH_REASON
        assert "2f" in BLOCKED_PATH_REASON
        assert "CCD" in BLOCKED_PATH_REASON


class TestDefocusProbePreflight:
    """离焦探针: 受阻 ⇒ 中止; 只是没聚焦 ⇒ 继续。"""

    def test_a_blocked_frame_raises(self):
        assessment = preflight_assessment(noise_frame())
        with pytest.raises(PathBlockedError) as excinfo:
            raise_if_path_blocked(assessment)
        assert "光束没有到达相机" in str(excinfo.value)

    def test_a_healthy_frame_does_not_raise(self):
        raise_if_path_blocked(preflight_assessment(spot_frame()))

    def test_a_frame_with_light_but_no_spot_does_not_raise(self):
        """光到了但没聚焦 —— 那一轮仍然有意义, 不许中止。"""
        broad = spot_frame(peak=20.0)
        broad = np.clip(broad - np.median(broad), 0.0, None)
        assessment = preflight_assessment(broad)
        assert assessment.blocked is False
        raise_if_path_blocked(assessment)

    def test_the_exit_code_is_its_own(self):
        """3 = dry-run 违约, 2 = 读帧超时 —— 光路受阻不能跟它们撞号。"""
        assert EXIT_PATH_BLOCKED not in (2, 3)


class TestDefocusProbePlan:
    """dry-run 的计划文本必须讲清楚这个 pre-flight。"""

    def test_the_plan_mentions_the_preflight(self):
        args = build_parser().parse_args(["--dry-run", "--camera-pixel-um", "2.2"])
        plan = build_plan(args, build_optics(args))
        preflight = plan["preflight"]
        assert "光路自检" in preflight["what"]
        assert "基线帧" in preflight["what"]
        assert "中止" in preflight["blocked"]
        assert "SLM" in preflight["checklist"]

    def test_the_dry_run_prints_the_preflight(self):
        proc = subprocess.run(
            # 像元在 dry-run 里必须显式给 (见探针的 assert_no_drivers): 读型号配置
            # 会 import 驱动包, 而 dry-run 保证不 import 任何 ao_shaping.drivers.*。
            [sys.executable, str(PROBE), "--dry-run", "--camera-pixel-um", "2.2"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "光路自检" in proc.stdout
        assert "pre-flight" in proc.stdout


class TestZernikeSweepPreflight:
    """泽尼克阶次扫描: 平场远场受阻 ⇒ 别跑 8 个阶次。"""

    def test_a_blocked_flat_frame_raises(self, monkeypatch):
        import ao_shaping.drivers.ccd.common as ccd_common

        from scripts.slm.run_gs_refine_zernike_sweep import probe_flat_field

        class _FakeCam:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_numpy_image(self, *args, **kwargs):
                return noise_frame((256, 256), seed=3)

        monkeypatch.setattr(
            ccd_common, "create_camera", lambda *a, **kw: _FakeCam()
        )
        with pytest.raises(SweepBlockedError) as excinfo:
            probe_flat_field("sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256)
        assert "光束没有到达相机" in str(excinfo.value)

    def test_a_healthy_flat_frame_still_probes(self, monkeypatch):
        import ao_shaping.drivers.ccd.common as ccd_common

        from scripts.slm.run_gs_refine_zernike_sweep import probe_flat_field

        class _FakeCam:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_numpy_image(self, *args, **kwargs):
                return spot_frame((256, 256))

        monkeypatch.setattr(
            ccd_common, "create_camera", lambda *a, **kw: _FakeCam()
        )
        center, w0, shape = probe_flat_field(
            "sim", cam_id=0, exposure_time_ms=0.0, seed=0, cam_size=256
        )
        assert shape == (256, 256)
        assert w0 > 0.0


class TestNoPartialArtefacts:
    """受阻时不能留下半份测量数据 —— 噪声行比没有行更坏。"""

    def test_the_assessment_is_json_ready(self):
        payload = assess_spot(noise_frame()).as_dict()
        assert payload["blocked"] is True
        assert payload["usable"] is False
        assert payload["reason"]
