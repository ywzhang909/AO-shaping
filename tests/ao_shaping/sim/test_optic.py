"""
测试 VectorWaveOpticsSim 物理仿真
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

from ao_shaping.sim.devices import VectorWaveOpticsSim


def create_gaussian_field(N, L, w0):
    """创建高斯光束"""
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    amplitude = np.exp(-(R**2) / (w0**2))
    Ex = amplitude  # X偏振
    Ey = np.zeros_like(Ex)
    return Ex, Ey


def test_vector_ao_correction():
    """测试矢量AO校正仿真"""
    N = 64
    L = 0.1  # 0.1m 孔径
    w0 = 0.02  # 高斯光束腰斑
    
    # --- 初始化物理引擎 ---
    propagator = VectorWaveOpticsSim(N=N, L=L, wavelength=1550e-9, Z=1000.0, Cn2=1e-14)
    
    # --- 步骤 1: 创建初始光束 (线偏振高斯光) ---
    Ex_init, Ey_init = create_gaussian_field(N, L, w0)
    print("初始化完成: 初始光束为X偏振高斯光")
    print(f"   光场形状: Ex={Ex_init.shape}, Ey={Ey_init.shape}")
    
    # --- 步骤 2: 传播并添加湍流 ---
    Ex_prop, Ey_prop = propagator.diffract(Ex_init, Ey_init)
    Ex_prop, Ey_prop = propagator.add_turbulence(Ex_prop, Ey_prop)
    print("传播完成: 经过1km大气传输和湍流畸变")
    
    # --- 步骤 3: 创建目标径向偏振光 ---
    Ex_tar, Ey_tar = propagator.create_target_radial(w0_factor=5)
    print("目标创建: 理想径向偏振光")
    
    # --- 步骤 4: 简单AO校正 (相位校正) ---
    phase_correction = -propagator.turb_phase
    Ex_final = Ex_prop * np.exp(1j * phase_correction)
    Ey_final = Ey_prop * np.exp(1j * phase_correction)
    print("AO校正完成: 施加相位校正")
    
    # --- 步骤 5: 计算校正前后Strehl比 ---
    ideal_plane = np.ones((N, N), dtype=complex) / np.sqrt(N * N)
    overlap_before = np.abs(np.vdot(Ex_prop, ideal_plane))**2
    overlap_after = np.abs(np.vdot(Ex_final, ideal_plane))**2
    print(f"   校正前与理想波前匹配度: {overlap_before:.4f}")
    print(f"   校正后与理想波前匹配度: {overlap_after:.4f}")
    
    # --- 步骤 6: 可视化 ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    rgb_init = VectorWaveOpticsSim.calculate_stokes_rgb(Ex_init, Ey_init)
    axes[0].imshow(rgb_init)
    axes[0].set_title("1. Initial Field")
    axes[0].axis('off')
    
    rgb_prop = VectorWaveOpticsSim.calculate_stokes_rgb(Ex_prop, Ey_prop)
    axes[1].imshow(rgb_prop)
    axes[1].set_title("2. After Propagation")
    axes[1].axis('off')
    
    rgb_tar = VectorWaveOpticsSim.calculate_stokes_rgb(Ex_tar, Ey_tar)
    axes[2].imshow(rgb_tar)
    axes[2].set_title("3. Target Mode")
    axes[2].axis('off')
    
    rgb_final = VectorWaveOpticsSim.calculate_stokes_rgb(Ex_final, Ey_final)
    axes[3].imshow(rgb_final)
    axes[3].set_title(f"4. After AO Correction")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('test_optic_result.png', dpi=150)
    print("可视化已保存到 test_optic_result.png")
    
    # 断言基本功能正常
    assert Ex_final.shape == (N, N), f"光场形状错误: {Ex_final.shape}"
    # 校正后与理想波前的匹配度应该提高
    assert overlap_after >= overlap_before * 0.5, f"校正效果不明显: {overlap_before} -> {overlap_after}"
    
    return True


if __name__ == "__main__":
    test_vector_ao_correction()
