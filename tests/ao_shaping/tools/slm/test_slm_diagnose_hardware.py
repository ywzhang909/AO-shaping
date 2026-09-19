"""SLM 硬件自检端到端 pytest 包装 (slm_diagnose 三步: freeze/modulate/linearity)。

在 2f Fourier 光路 (SLM 前焦面 -> f=125mm 透镜 -> CCD 后焦面) 下对
Santec SLM + MiiCam 做端到端自检, 覆盖 ``slm_diagnose.py`` 的独立步骤函数。

用法 (SLM + MiiCam 硬件在线时)::

    AO_RUN_HARDWARE=1 python -m pytest -m hardware tests/ao_shaping/tools/slm/test_slm_diagnose_hardware.py -v

- 模块级 ``pytestmark = pytest.mark.hardware``: 常规 ``pytest`` 不收集/不运行,
  需显式 ``-m hardware`` (pyproject.toml 已注册该 marker)。
- 函数级 ``AO_RUN_HARDWARE=1`` 门控: 未设置则 ``pytest.skip``, 防止误触 SLM
  硬件 (尤其防 DVI 挂起类风险)。
- 语义: 断言每步**可运行且返回结构化 DiagnoseResult** (metrics 键齐全,
  ``ok`` 为 bool); 不做 ``ok`` 断言 —— 面板真故障时 ``ok=False`` 正是自检
  要暴露的合法诊断结果。

设备打开方式与 ``slm_diagnose.main`` 同源: ``Santec(slm_number, wavelength,
video_mode=0)`` + ``utils.slm_camera.open_miicam_camera(cam_id, exposure_ms)``,
仅使用 memory 模式 (video_mode=0), 绝不自动尝试 DVI。
"""

from __future__ import annotations

import os

import pytest

from ao_shaping.tools.slm.slm_diagnose import (
    DiagnoseResult,
    step_freezing,
    step_linearity,
    step_modulation,
)

pytestmark = pytest.mark.hardware

# 与 slm_diagnose.main 的 CLI 参数一致 (除 slm_wavelength: CLI 默认 1064 为
# 历史基准, 实机为 532nm; Santec 按 wavelength 动态查询 2π 对应灰度).
_SLM_NUMBER = 1
_SLM_WAVELENGTH_NM = 532
_CAM_ID = 0
_PERIOD_REF = 64
_PERIOD_TEST = 32
_EXPOSURE_MS = 2.0
_SETTLE_S = 1.0

_ENV = "AO_RUN_HARDWARE"


def _require_hardware() -> None:
    """环境门控: ``AO_RUN_HARDWARE=1`` 未设置则跳过硬件测试。"""
    if os.environ.get(_ENV, "").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip(f"set {_ENV}=1 to run SLM hardware self-check tests")


@pytest.fixture(scope="module")
def devices():
    """打开 Santec SLM (memory 模式) + MiiCam; 语义与 slm_diagnose.main 一致。

    Yield ``(slm, camera)``; teardown 逆序关闭 (先相机后 SLM, 各自 try/except
    守护, 与 CLI 的 finally 块相同)。
    """
    _require_hardware()
    from ao_shaping.drivers.ccd import MIICamera
    from ao_shaping.drivers.slm.santec import Santec

    slm = Santec(slm_number=_SLM_NUMBER, wavelength=_SLM_WAVELENGTH_NM, video_mode=0)
    slm.open()
    camera = None
    try:
        camera = MIICamera(_CAM_ID, _EXPOSURE_MS).open()
        yield slm, camera
    finally:
        try:
            if camera is not None:
                camera.close()
        finally:
            slm.close()


def _assert_result(result: DiagnoseResult, metric_key: str) -> None:
    """结构化结果校验: 类型/ok 是 bool/message 非空/指标键存在。"""
    assert isinstance(result, DiagnoseResult)
    assert isinstance(result.ok, bool)
    assert result.message
    assert metric_key in result.metrics


def test_freezing_step_runs_and_returns_structure(devices):
    """patches 冻结检测: 可运行且返回结构化结果 (不断言 ok 值)。"""
    slm, camera = devices
    result = step_freezing(
        slm,
        camera,
        period_ref=_PERIOD_REF,
        period_test=_PERIOD_TEST,
        slm_wavelength=_SLM_WAVELENGTH_NM,
        settle_s=_SETTLE_S,
        exposure_ms=_EXPOSURE_MS,
    )
    _assert_result(result, "n_different")


def test_modulation_step_runs_and_returns_structure(devices):
    """灰度调制检测: 可运行且返回结构化结果 (不断言 ok 值)。"""
    slm, camera = devices
    result = step_modulation(
        slm,
        camera,
        slm_wavelength=_SLM_WAVELENGTH_NM,
        settle_s=_SETTLE_S,
        exposure_ms=_EXPOSURE_MS,
    )
    _assert_result(result, "rel_spread")


def test_linearity_step_runs_and_returns_structure(devices):
    """曝光->亮度线性检测: 可运行且返回结构化结果 (不断言 ok 值)。"""
    slm, camera = devices
    result = step_linearity(
        slm,
        camera,
        period_ref=_PERIOD_REF,
        slm_wavelength=_SLM_WAVELENGTH_NM,
        settle_s=_SETTLE_S,
        base_exposure_ms=_EXPOSURE_MS,
    )
    _assert_result(result, "growth_x20")