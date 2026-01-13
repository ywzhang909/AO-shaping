"""
湍流条件分析测试脚本

测试不同湍流强度对AO系统性能的影响
"""

import numpy as np
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ao_shaping.sim.devices import AOConfig, TraditionalAOSystem
from ao_shaping.utils.spots_calc import centroid, effective_radius, calculate_sharpness, power_bucket, make_coord


def analyze_turbulence_effects():
    """分析不同湍流强度对AO系统的影响"""

    # 不同湍流强度 (Cn2值)
    cn2_values = [0.0, 1e-15, 1e-14, 5e-14, 1e-13, 5e-13]

    results = []

    for cn2 in cn2_values:
        print(f"\n测试湍流强度 Cn2 = {cn2:.2e}")

        # 配置AO系统
        config = AOConfig(N=128, Cn2=cn2, dm_actuators=8, subapertures=8)
        system = TraditionalAOSystem(config)

        # 获取图像
        image = system.get_image().astype(float)
        slopes = system.measure_wavefront()

        # 调试信息
        r0 = system.turbulence._calculate_fried_parameter()
        phase_screen_rms = np.sqrt(np.mean(system.turbulence.phase_screen**2))
        print(f"  Fried参数 r0: {r0:.6f}")
        print(f"  湍流相位屏RMS: {phase_screen_rms:.6f}")
        print(f"  最终相位RMS: {np.sqrt(np.mean(np.angle(system.E_corrected)**2)):.6f}")

        # 计算spots_calc指标
        cx, cy = centroid(image)
        radius = effective_radius(image, dpix=0.1, clip=0.5)
        # 计算锐度（改进版本，避免数值问题）
        img_float = image.astype(float)
        if img_float.max() > 0:
            img_normalized = img_float / img_float.max()
            gradient_x = np.gradient(img_normalized, axis=1)
            gradient_y = np.gradient(img_normalized, axis=0)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            sharpness = np.mean(gradient_magnitude)
        else:
            sharpness = 0.0

        xv, yv = make_coord(image)
        center = image.shape[0] // 2
        power_10 = power_bucket(image, xv, yv, center=(center, center), r_bucket=10)
        power_20 = power_bucket(image, xv, yv, center=(center, center), r_bucket=20)

        # 系统性能指标
        strehl = system.reset()['strehl']
        total_power = np.sum(image)

        result = {
            'cn2': cn2,
            'centroid_x': cx,
            'centroid_y': cy,
            'effective_radius': radius,
            'sharpness': sharpness,
            'power_r10': power_10,
            'power_r20': power_20,
            'strehl_ratio': strehl,
            'total_power': total_power,
            'rms_slopes': np.sqrt(np.mean(slopes**2))
        }

        results.append(result)
        print(".4f"
              ".2f"
              ".2f"
              ".2f")

    return results


def analyze_zernike_reconstruction():
    """分析Zernike相位重建的一致性"""

    from ao_shaping.sim.devices import ZernikePolynomials

    config = AOConfig(N=128, subapertures=8)
    system = TraditionalAOSystem(config)

    # 测试不同Zernike模式
    modes_to_test = [1, 2, 4, 6, 8]  # Tilt X, Tilt Y, Defocus, Astigmatism, etc.
    mode_names = ['Tilt X', 'Tilt Y', 'Defocus', 'Astigmatism 45°', 'Trefoil X']

    reconstruction_results = []

    for i, mode_idx in enumerate(modes_to_test):
        basis = ZernikePolynomials.generate_basis(mode_idx + 3, config.N, 2.0)
        input_phase = basis[mode_idx] * 1.0  # 单位幅度

        # 创建电场
        E = system.E_in * np.exp(1j * input_phase)
        intensity = np.abs(E)**2

        # 测量斜率
        slopes = system.wfs.measure_slopes(intensity, np.angle(E))

        # 重建
        reconstructed = system.wfs.reconstruct_wavefront(slopes, basis[:mode_idx+3])

        # 计算相关性和RMS误差
        correlation = np.corrcoef(input_phase.flatten(), reconstructed.flatten())[0, 1]
        rms_error = np.sqrt(np.mean((input_phase - reconstructed)**2))

        result = {
            'mode': mode_names[i],
            'mode_idx': mode_idx,
            'correlation': correlation,
            'rms_error': rms_error
        }

        reconstruction_results.append(result)
        print(f"{mode_names[i]} (模式{mode_idx}): 相关性={correlation:.3f}, RMS误差={rms_error:.3f}")

    return reconstruction_results


if __name__ == "__main__":
    print("=== AO系统湍流影响分析 ===")
    turbulence_results = analyze_turbulence_effects()

    print("\n=== Zernike重建一致性分析 ===")
    reconstruction_results = analyze_zernike_reconstruction()

    # 保存结果用于markdown生成
    import json
    output = {
        'turbulence_analysis': turbulence_results,
        'reconstruction_analysis': reconstruction_results
    }

    with open('turbulence_test_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n结果已保存到 turbulence_test_results.json")