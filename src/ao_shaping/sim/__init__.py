"""
AO仿真模块

包含自适应光学系统各组件的仿真实现。
"""

from .devices import (
    LightSource,
    DeformableMirror,
    AtmosphericTurbulence,
    HartmannShackWavefrontSensor,
    Camera,
    VectorWavePropagator,
    TraditionalAOSystem,
    AOConfig,
    VectorWaveOpticsSim,
    zernike_phase_screen,
    calculate_strehl,
    calculate_rms,
    calculate_pv,
    # 别名
    LaserSource,
    HartmannSensor,
    TurbulencePhaseScreen,
    CCD,
    OpticalPropagator,
    AOSystem,
    CCDCamera,
    WavefrontPropagator,
    HartmannShackSensor,
)

__all__ = [
    'LightSource',
    'DeformableMirror',
    'AtmosphericTurbulence',
    'HartmannShackWavefrontSensor',
    'Camera',
    'VectorWavePropagator',
    'TraditionalAOSystem',
    'AOConfig',
    'VectorWaveOpticsSim',
    'zernike_phase_screen',
    'calculate_strehl',
    'calculate_rms',
    'calculate_pv',
    # 别名
    'LaserSource',
    'HartmannSensor',
    'TurbulencePhaseScreen',
    'CCD',
    'OpticalPropagator',
    'AOSystem',
    'CCDCamera',
    'WavefrontPropagator',
    'HartmannShackSensor',
]
