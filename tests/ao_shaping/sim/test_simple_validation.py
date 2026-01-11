"""
简单验证测试 - 检查各个组件的基本功能
"""

import numpy as np
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ao_shaping.sim.devices import AtmosphericTurbulence, ZernikePolynomials

def test_turbulence_generation():
    """测试湍流生成"""
    print("=== 测试湍流生成 ===")

    # 设置随机种子确保一致性
    np.random.seed(42)

    # 测试不同Cn2值
    cn2_values = [0.0, 1e-14, 1e-13]

    for cn2 in cn2_values:
        turb = AtmosphericTurbulence(Cn2=cn2, N=64, L=0.1, seed=42)
        phase_screen = turb.get_phase_screen()

        rms = np.sqrt(np.mean(phase_screen**2))
        print(".2e")

        # 检查基本属性
        assert phase_screen.shape == (64, 64), f"形状错误: {phase_screen.shape}"
        assert np.isfinite(rms), f"RMS不是有限值: {rms}"

def test_zernike_basis():
    """测试Zernike基函数"""
    print("\n=== 测试Zernike基函数 ===")

    basis = ZernikePolynomials.generate_basis(10, 64, 2.0)
    print(f"基函数形状: {basis.shape}")

    # 检查基本属性
    assert basis.shape[0] >= 10, f"模式数量不足: {basis.shape[0]}"
    assert basis.shape[1] == 64, f"网格尺寸错误: {basis.shape[1]}"

    # 检查正交性（简化检查）
    for i in range(min(5, basis.shape[0])):
        norm = np.sqrt(np.mean(basis[i]**2))
        print(".3f")

def test_correlation_calculation():
    """测试相关性计算"""
    print("\n=== 测试相关性计算 ===")

    # 创建测试数据
    np.random.seed(42)
    a = np.random.randn(100)
    b = a + 0.1 * np.random.randn(100)  # 高度相关

    corr = np.corrcoef(a, b)[0, 1]
    print(".3f")

    # 测试边缘情况
    c = np.random.randn(100)  # 不相关
    corr_unrelated = np.corrcoef(a, c)[0, 1]
    print(".3f")

    # 测试常数
    d = np.ones(100)
    try:
        corr_const = np.corrcoef(a, d)[0, 1]
        print(".3f")
    except:
        print("常数相关性计算失败（预期）")

def test_ao_system_integration():
    """测试AO系统集成"""
    print("\n=== 测试AO系统集成 ===")

    from ao_shaping.sim.devices import TraditionalAOSystem, AOConfig

    # 测试无湍流
    config = AOConfig(N=64, Cn2=0.0)
    system = TraditionalAOSystem(config)

    print("无湍流系统:")
    print(f"  湍流相位屏RMS: {np.sqrt(np.mean(system.turbulence.phase_screen**2)):.6f}")
    print(f"  输入场RMS: {np.sqrt(np.mean(np.abs(system.E_in)**2)):.6f}")
    print(f"  湍流后场RMS: {np.sqrt(np.mean(np.abs(system.E_turb)**2)):.6f}")
    print(f"  传播后场RMS: {np.sqrt(np.mean(np.abs(system.E_propagated)**2)):.6f}")
    print(f"  校正后场RMS: {np.sqrt(np.mean(np.abs(system.E_corrected)**2)):.6f}")

    # 测试有湍流 - 直接创建湍流对象
    print("\n直接测试湍流对象:")
    from ao_shaping.sim.devices import AtmosphericTurbulence
    turb_direct = AtmosphericTurbulence(Cn2=1e-14, N=64, L=0.1)
    print(f"  直接创建湍流RMS: {np.sqrt(np.mean(turb_direct.phase_screen**2)):.6f}")

    # 测试有湍流 - 通过AO系统
    config_turb = AOConfig(N=64, Cn2=1e-14)
    system_turb = TraditionalAOSystem(config_turb)

    print("\n有湍流系统:")
    print(f"  Cn2配置值: {system_turb.config.Cn2}")
    print(f"  湍流对象Cn2: {system_turb.turbulence.Cn2}")
    print(f"  湍流网格尺寸: {system_turb.turbulence.N}")
    print(f"  湍流相位屏RMS: {np.sqrt(np.mean(system_turb.turbulence.phase_screen**2)):.6f}")

    # 手动检查湍流应用
    test_field = np.ones((64, 64), dtype=complex)
    turb_field = system_turb.turbulence.add_phase_screen(test_field)
    phase_rms = np.sqrt(np.mean(np.angle(turb_field)**2))
    print(f"  手动湍流应用相位RMS: {phase_rms:.6f}")

    print(f"  输入场RMS: {np.sqrt(np.mean(np.abs(system_turb.E_in)**2)):.6f}")
    print(f"  湍流后场RMS: {np.sqrt(np.mean(np.abs(system_turb.E_turb)**2)):.6f}")
    print(f"  传播后场RMS: {np.sqrt(np.mean(np.abs(system_turb.E_propagated)**2)):.6f}")
    print(f"  校正后场RMS: {np.sqrt(np.mean(np.abs(system_turb.E_corrected)**2)):.6f}")

if __name__ == "__main__":
    test_turbulence_generation()
    test_zernike_basis()
    test_correlation_calculation()
    test_ao_system_integration()
    print("\n=== 验证完成 ===")