"""FourierGSNet 仿真环境的物理回归测试 (TDD)。

验证 ``SimFourierGSNetEnv`` 的 2f 傅里叶台架物理模型:

* **K 定律** —— 周期 ``P_slm`` 个 SLM 像素的闪耀光栅, 其 +1 级恰好落在距 0 级
  ``K_px / P_slm`` 个 CCD 像素处;
* **物理像元包络** —— 可分离 sinc, 首零点位于 ``K_px * d_slm / d_eff`` 个 CCD 像素;
* **静态像差可加性** —— ``env.aberrations`` 等价于直接向 ``display_phase`` 写
  ``base + aberration``, 且 Noll 3 (倾斜) 使远场质心偏移 ``c * K / (pi * R)`` 像素;
* **鸭子类型契约** —— ``SimSLM``/``SimCCD`` 满足真实 Santec/MiiCam 驱动的调用形状
  (独立 CLI ``fouriergsnet_optimize.py`` 已于 f19dced 删除, 契约本身仍有效)。

全部测试可设种子、纯 CPU、K 取小值 (2048), 且用 ``noise_enabled=False`` 让物理
部分完全确定性。

与其他模块的关系: 被测对象是 ``drivers/sim/fouriergsnet_env.py``; 其 FFT 口径被
``algorithm/signal_processing/zernike_coefficient_optimizer.py`` 的正向模型共享,
故这里的 K 定律/包络断言同时是那条 model-in-the-loop 拟合链的前置条件。
"""
from __future__ import annotations

import numpy as np

from ao_shaping.drivers.sim.fouriergsnet_env import BeamParams, SimFourierGSNetEnv
from ao_shaping.utils.wavefront.zernike_calc import ZernikeGenerator

K_SMALL = 2048
D_SLM = 8e-6
D_EFF_TEST = 4 * D_SLM  # 包络首零点落在带内 K_px/4 = 512 px
PANEL_H, PANEL_W = 1920, 1200


def _blaze_phase(period_px: int) -> np.ndarray:
    """沿 x 方向的闪耀光栅相位 (弧度), 复刻历史管线中 SLMCCDCalibrator._blaze。"""
    return np.tile((2 * np.pi * np.arange(PANEL_W) / period_px), (PANEL_H, 1))


def _threshold_centroid(img: np.ndarray, thresh: float = 0.5) -> tuple[float, float]:
    """高于 ``thresh * max`` 的像素的强度加权质心。"""
    yy, xx = np.nonzero(img > thresh * img.max())
    w = img[yy, xx].astype(np.float64)
    return (float(np.average(yy, weights=w)), float(np.average(xx, weights=w)))


def test_k_law(seed: int = 0) -> None:
    """光栅周期 P_slm -> +1 级位移 == K_px / P_slm (容差 3%)。"""
    env = SimFourierGSNetEnv(K_px=K_SMALL, noise_enabled=False, seed=seed)
    center = K_SMALL / 2
    for period in (32, 64):
        env.slm.display_phase(_blaze_phase(period))
        frame = env.ccd.get_numpy_image(n_sample=1)
        _, col = _threshold_centroid(frame)
        disp = abs(col - center)
        expected = K_SMALL / period
        assert abs(disp - expected) / expected < 0.03, (
            f"period={period}: disp={disp:.3f}, expected={expected:.3f}"
        )


def test_sinc_envelope_null(seed: int = 0) -> None:
    """平场: 可分离 sinc 包络首零点位于 K_px * d_slm / d_eff (容差 5%)。"""
    env = SimFourierGSNetEnv(
        K_px=K_SMALL,
        d_eff=D_EFF_TEST,
        beam=BeamParams(w0=1.0),  # 准点状光束 -> 远场 ≈ 包络本身
        noise_enabled=False,
        seed=seed,
    )
    env.slm.display_phase(np.zeros((PANEL_H, PANEL_W)))
    frame = env.ccd.get_numpy_image(n_sample=1)
    center = K_SMALL // 2
    profile = frame[center, :].astype(np.float64)
    null_offset = 20 + int(np.argmin(profile[center + 20:]))
    expected = K_SMALL * D_SLM / D_EFF_TEST  # 512 px
    assert abs(null_offset - expected) / expected < 0.05, (
        f"null at {null_offset} px, expected {expected} px"
    )


def test_static_aberration_additivity(seed: int = 0) -> None:
    """静态像差叠加到已显示基础相位上 (env 路径 == 直接路径)。"""
    region = 512
    zgen = ZernikeGenerator((region, region), radius=region / 2, n_orders=6)
    base = zgen.generate_polynomial({(2, 0): 0.3})  # 小离焦
    tilt = zgen.generate_polynomial({(1, -1): 1.0})  # Noll 3 = (1,-1), 沿行倾斜
    r0, c0 = 960 - region // 2, 600 - region // 2

    env = SimFourierGSNetEnv(
        K_px=K_SMALL, beam=BeamParams(w0=100.0), noise_enabled=False, seed=seed
    )
    panel_base = np.zeros((PANEL_H, PANEL_W))
    panel_base[r0:r0 + region, c0:c0 + region] = base
    env.slm.display_phase(panel_base)
    env.aberrations = {3: 1.0}  # Noll 3
    frame_env = env.ccd.get_numpy_image(n_sample=1)

    env_ref = SimFourierGSNetEnv(
        K_px=K_SMALL, beam=BeamParams(w0=100.0), noise_enabled=False, seed=seed
    )
    panel_ref = np.zeros((PANEL_H, PANEL_W))
    panel_ref[r0:r0 + region, c0:c0 + region] = base + tilt
    env_ref.slm.display_phase(panel_ref)
    frame_ref = env_ref.ccd.get_numpy_image(n_sample=1)

    assert np.allclose(frame_env, frame_ref, rtol=1e-2)

    # 倾斜引起的质心偏移: c * K / (πR) = 2.55 px (c=1, R=256, K=2048)
    env_flat = SimFourierGSNetEnv(
        K_px=K_SMALL, beam=BeamParams(w0=100.0), noise_enabled=False, seed=seed
    )
    env_flat.slm.display_phase(np.zeros((PANEL_H, PANEL_W)))
    frame_flat = env_flat.ccd.get_numpy_image(n_sample=1)
    cy_env, _ = _threshold_centroid(frame_env)
    cy_flat, _ = _threshold_centroid(frame_flat)
    shift = cy_env - cy_flat
    expected = 1.0 * K_SMALL / (np.pi * (region / 2))
    assert abs(shift - expected) / expected < 0.2, (
        f"tilt shift {shift:.3f} px, expected {expected:.3f} px"
    )


def test_device_ducktyping_contract(seed: int = 0) -> None:
    """SimSLM/SimCCD 满足真实 SLM/MiiCam 驱动的鸭子类型契约。"""
    env = SimFourierGSNetEnv(K_px=K_SMALL, seed=seed)
    slm = env.slm
    ccd = env.ccd

    assert slm.MEMORY_MODE_INTERNAL == 0
    assert slm.Panel_Res == (1920, 1200)

    ret = slm.display_phase(
        np.zeros((PANEL_H, PANEL_W)), wait_time_s=0.1, memory_number=3, memory_mode=0
    )
    assert isinstance(ret, int)
    assert len(slm.phase_calls) == 1
    call = slm.phase_calls[0]
    assert call["wait_time_s"] == 0.1
    assert call["memory_number"] == 3
    assert call["memory_mode"] == 0

    ret2 = slm.display_data(np.zeros((PANEL_H, PANEL_W), dtype=np.uint16), wait_time_s=0.1)
    assert isinstance(ret2, int)
    assert len(slm.data_calls) == 1

    assert isinstance(slm.get_displayed_memory_number(), int)

    frame = ccd.get_numpy_image(n_sample=1)
    assert frame.dtype == np.uint16
    assert frame.shape == (K_SMALL, K_SMALL)